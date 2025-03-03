import os
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from ..utils.logger import RichLogger
from ..config.config import config_class as config
from ..utils.timing import timeit
from ..utils.environment import create_directory
from ..mitsuba.camera import CameraUtils

logger = RichLogger.get_logger("mitsuba_app.processing.scene")

class SceneProcessor:
    """
    Processor for creating and managing scene files.
    """

    @staticmethod
    def ensure_default_template():
        """
        Ensure the default template file exists.
        """
        if not os.path.exists(config.TEMPLATE_XML):
            # Try to create the directory if it doesn't exist
            template_dir = os.path.dirname(config.TEMPLATE_XML)
            os.makedirs(template_dir, exist_ok=True)
            
            # Create a basic template if it doesn't exist
            with open(config.TEMPLATE_XML, 'w') as f:
                f.write('''<?xml version="1.0"?>
<scene version="2.0.0">
    <integrator type="path">
        <integer name="max_depth" value="16"/>
    </integrator>

    <sensor type="perspective">
        <float name="fov" value="45"/>
        <transform name="to_world">
            <lookat origin="0, 0, 4" target="0, 0, 0" up="0, 1, 0"/>
        </transform>
        <film type="hdrfilm">
            <integer name="width" value="1920"/>
            <integer name="height" value="1080"/>
        </film>
        <sampler type="independent">
            <integer name="sample_count" value="2048"/>
        </sampler>
    </sensor>

    <bsdf type="diffuse" id="default-bsdf">
        <rgb name="reflectance" value="0.8, 0.8, 0.8"/>
    </bsdf>

    <shape type="obj">
        <string name="filename" value="<!-- OBJ_FILENAME -->"/>
        <ref id="default-bsdf"/>
    </shape>

    <emitter type="constant">
        <rgb name="radiance" value="1.0, 1.0, 1.0"/>
    </emitter>
</scene>''')
        return config.TEMPLATE_XML

    @classmethod
    @timeit(log_level="debug")
    def create_scene_xml_for_obj(
        cls, 
        obj_file: str, 
        output_xml_path: str, 
        obj_path_in_xml: Optional[str] = None,
        camera_view: str = "perspective"
    ) -> str:
        """
        Create a Mitsuba scene XML file for an OBJ file.
        
        Args:
            obj_file: Path to OBJ file
            output_xml_path: Path for output XML file
            obj_path_in_xml: Path to use in the XML (defaults to obj_file)
            camera_view: Camera view to use (perspective, front, top, etc.)
            
        Returns:
            Path to created XML file
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_xml_path), exist_ok=True)
        
        # Read template
        template_path = config.TEMPLATE_XML
        with open(template_path, 'r') as file:
            xml_content = file.read()
            
        # Set OBJ path in XML
        obj_path_to_use = obj_path_in_xml if obj_path_in_xml else obj_file
        xml_content = xml_content.replace("<!-- OBJ_FILENAME -->", obj_path_to_use)
        
        try:
            # Apply camera view
            xml_content = cls._apply_camera_view(xml_content, camera_view)
            
            # Apply AOV settings if enabled
            if config.AOV_ENABLED and hasattr(config, 'AOV_STRING') and config.AOV_STRING:
                xml_content = cls._apply_aov_settings(xml_content, config.AOV_STRING)
        except Exception as e:
            logger.error(f"Error applying camera view or AOV settings: {e}")
            # Continue with default settings
        
        # Save the XML file
        with open(output_xml_path, 'w') as file:
            file.write(xml_content)
            
        return output_xml_path

    @classmethod
    @timeit(log_level="debug") 
    def create_scene_xmls_with_multiple_views(
        cls, 
        obj_file: str, 
        output_base_path: str, 
        views: List[str],
        obj_path_in_xml: Optional[str] = None,
        render_params: Dict[str, Any] = None
    ) -> Dict[str, str]:
        """
        Create a Mitsuba scene XML file for each view.
        
        Args:
            obj_file: Path to OBJ file
            output_base_path: Base path for output XML files
            views: List of camera views to create XMLs for
            obj_path_in_xml: Path to use in the XML (defaults to obj_file)
            render_params: Optional rendering parameters to apply
            
        Returns:
            Dictionary mapping view names to XML file paths
        """
        result = {}
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_base_path), exist_ok=True)
        
        # Create XML for each view
        for view in views:
            output_xml_path = f"{output_base_path}_{view}.xml"
            try:
                xml_path = cls.create_scene_xml_for_obj(
                    obj_file=obj_file,
                    output_xml_path=output_xml_path,
                    obj_path_in_xml=obj_path_in_xml,
                    camera_view=view
                )
                result[view] = xml_path
            except Exception as e:
                logger.error(f"Error creating scene XML for view '{view}': {e}")
            
        return result
    
    @classmethod
    def _apply_camera_view(cls, xml_content: str, view: str) -> str:
        """
        Apply a camera view to an XML template.
        
        Args:
            xml_content: XML content to modify
            view: Camera view to apply
            
        Returns:
            Modified XML content
        """
        # Default view remains unchanged
        if view == "perspective":
            logger.debug("Using default perspective view (no changes)")
            return xml_content
        
        try:
            # Parse XML
            root = ET.fromstring(xml_content)
            
            # Get camera settings for the view
            logger.debug(f"Getting camera settings for view: '{view}'")
            camera_settings = CameraUtils.get_camera_settings_for_view(view)
            logger.debug(f"Camera settings: origin='{camera_settings['origin']}', target='{camera_settings['target']}', up='{camera_settings['up']}'")
            
            # Find the sensor element
            sensor = root.find(".//sensor")
            if sensor is not None:
                # Find the transform element
                transform = sensor.find(".//transform")
                if transform is not None:
                    # Check if we have a lookat element
                    lookat = transform.find("lookat")
                    
                    if lookat is not None:
                        # Update existing lookat parameters
                        old_origin = lookat.get("origin", "unknown")
                        old_target = lookat.get("target", "unknown")
                        old_up = lookat.get("up", "unknown")
                        
                        lookat.set("origin", camera_settings["origin"])
                        lookat.set("target", camera_settings["target"])
                        lookat.set("up", camera_settings["up"])
                        
                        logger.debug(f"Updated camera transform from: origin='{old_origin}', target='{old_target}', up='{old_up}'")
                        logger.debug(f"                           to: origin='{camera_settings['origin']}', target='{camera_settings['target']}', up='{camera_settings['up']}'")
                    else:
                        # No lookat element - we need to replace the entire transform
                        logger.debug("No lookat element found, replacing entire transform")
                        
                        # Clear existing transform content (remove rotate/translate elements)
                        for child in list(transform):
                            transform.remove(child)
                        
                        # Add a new lookat element
                        ET.SubElement(transform, "lookat", {
                            "origin": camera_settings["origin"],
                            "target": camera_settings["target"],
                            "up": camera_settings["up"]
                        })
                        
                        logger.debug(f"Created new lookat transform with origin='{camera_settings['origin']}', target='{camera_settings['target']}', up='{camera_settings['up']}'")
                else:
                    logger.warning("No transform element found in camera settings")
            else:
                logger.warning("No sensor element found in XML")
                
            # Convert back to string with XML declaration
            xml_declaration = '<?xml version="1.0" encoding="utf-8"?>\n'
            result = xml_declaration + ET.tostring(root, encoding='utf-8').decode('utf-8')
            logger.debug(f"Successfully applied '{view}' camera view")
            return result
            
        except Exception as e:
            logger.error(f"Error applying camera view '{view}': {e}", exc_info=True)
            return xml_content

    @classmethod
    def _apply_aov_settings(cls, xml_content: str, aov_string: str) -> str:
        """
        Apply AOV settings to an XML template.
        
        Args:
            xml_content: XML content to modify
            aov_string: AOV string (e.g., "albedo:albedo,nn:sh_normal")
            
        Returns:
            Modified XML content with AOV integrator configured
        """
        try:
            # Parse XML
            root = ET.fromstring(xml_content)
            
            # Find the original integrator
            original_integrator = root.find(".//integrator")
            if original_integrator is not None:
                # Get the original integrator's attributes and children
                integrator_type = original_integrator.get("type", "path")
                integrator_children = list(original_integrator)
                
                # Create a new AOV integrator
                aov_integrator = ET.Element("integrator", {"type": "aov"})
                
                # Add AOVs string parameter
                aov_string_elem = ET.SubElement(aov_integrator, "string", {"name": "aovs", "value": aov_string})
                
                # Create nested integrator
                nested_integrator = ET.SubElement(aov_integrator, "integrator", {"type": integrator_type, "name": "main"})
                
                # Copy original integrator's children to nested integrator
                for child in integrator_children:
                    nested_integrator.append(child)
                    
                # Replace original integrator with AOV integrator
                parent = root
                for i, child in enumerate(parent):
                    if child.tag == "integrator":
                        parent[i] = aov_integrator
                        break
                
            # Convert back to string
            return ET.tostring(root, encoding='utf-8', xml_declaration=True).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error applying AOV settings: {e}")
            return xml_content

    @classmethod
    def update_template_with_params(
        cls, 
        template_path: str,
        spp: int = 256,
        width: int = 1920, 
        height: int = 1080,
        max_depth: int = 8
    ) -> bool:
        """
        Update the template XML with custom rendering parameters.
        
        Args:
            template_path: Path to template XML
            spp: Samples per pixel
            width: Image width
            height: Image height
            max_depth: Maximum ray depth
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read the template as plain text to preserve formatting
            with open(template_path, 'r') as f:
                xml_content = f.read()
            
            # Update the parameters using regex for safer pattern matching
            xml_content = re.sub(r'<default name="spp" value="[^"]*"', 
                                f'<default name="spp" value="{spp}"', xml_content)
            xml_content = re.sub(r'<default name="resx" value="[^"]*"', 
                                f'<default name="resx" value="{width}"', xml_content)
            xml_content = re.sub(r'<default name="resy" value="[^"]*"', 
                                f'<default name="resy" value="{height}"', xml_content)
            xml_content = re.sub(r'<default name="max_depth" value="[^"]*"', 
                                f'<default name="max_depth" value="{max_depth}"', xml_content)
            
            # Write the modified XML back
            with open(template_path, 'w') as f:
                f.write(xml_content)
            
            # Log parameters
            logger.debug(f"Updated template {template_path} with parameters: spp={spp}, width={width}, height={height}, max_depth={max_depth}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating template with parameters: {e}")
            return False
