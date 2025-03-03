"""
XML processing utilities for the Mitsuba application.
"""

import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, Optional
import xmlschema
from src.utils.logger import RichLogger
from src.config.config import config_class as config
from src.utils.timing import timeit

logger = RichLogger.get_logger("mitsuba_app.processing.xml")

class XMLValidator:
    """
    Utility class for XML validation against schemas.
    """
    
    @staticmethod
    @timeit(log_level="debug")
    def validate_xml(xml_path: str, xsd_path: str) -> bool:
        """
        Validate an XML file against the provided XSD schema.
        
        Args:
            xml_path: Path to XML file
            xsd_path: Path to XSD schema file
            
        Returns:
            True if XML is valid, False otherwise
        """
        try:
            schema = xmlschema.XMLSchema(xsd_path)
            is_valid = schema.is_valid(xml_path)
            if is_valid:
                logger.debug(f"XML '{xml_path}' is valid according to '{xsd_path}'.")
            else:
                logger.error(f"XML '{xml_path}' does not conform to the schema '{xsd_path}'.")
            return is_valid
        except Exception as e:
            logger.error(f"An error occurred while validating XML: {e}")
            return False

class XMLProcessor:
    """
    Utility class for XML file processing.
    """
    
    @staticmethod
    @timeit(log_level="debug")
    def create_basic_scene_xml(output_path: str, 
                              shape_file: Optional[str] = None,
                              spp: int = 256,
                              width: int = 1920,
                              height: int = 1080,
                              max_depth: int = 16) -> str:
        """
        Create a basic Mitsuba scene XML file from scratch.
        
        Args:
            output_path: Path to write the XML file
            shape_file: Path to a shape file (OBJ, etc.)
            spp: Samples per pixel
            width: Output width in pixels
            height: Output height in pixels
            max_depth: Maximum ray depth
            
        Returns:
            Path to the created XML file
        """
        try:
            # Create the root element
            root = ET.Element("scene", {"version": "2.1.0"})
            
            # Add defaults
            ET.SubElement(root, "default", {"name": "spp", "value": str(spp)})
            ET.SubElement(root, "default", {"name": "resx", "value": str(width)})
            ET.SubElement(root, "default", {"name": "resy", "value": str(height)})
            
            # Add integrator
            integrator = ET.SubElement(root, "integrator", {"type": "path"})
            ET.SubElement(integrator, "integer", {"name": "max_depth", "value": str(max_depth)})
            
            # Add sensor (camera)
            sensor = ET.SubElement(root, "sensor", {"type": "perspective"})
            ET.SubElement(sensor, "string", {"name": "fov_axis", "value": "x"})
            ET.SubElement(sensor, "float", {"name": "fov", "value": "39.597755"})
            
            # Add camera transform
            transform = ET.SubElement(sensor, "transform", {"name": "to_world"})
            ET.SubElement(transform, "rotate", {"x": "1", "angle": "-153.559291"})
            ET.SubElement(transform, "rotate", {"y": "1", "angle": "-46.691938"})
            ET.SubElement(transform, "rotate", {"z": "1", "angle": "-179.999991"})
            ET.SubElement(transform, "translate", {"value": "7.358891 4.958309 6.925791"})
            
            # Add sampler
            sampler = ET.SubElement(sensor, "sampler", {"type": "independent", "name": "sampler"})
            ET.SubElement(sampler, "integer", {"name": "sample_count", "value": "$spp"})
            
            # Add film
            film = ET.SubElement(sensor, "film", {"type": "hdrfilm", "name": "film"})
            ET.SubElement(film, "integer", {"name": "width", "value": "$resx"})
            ET.SubElement(film, "integer", {"name": "height", "value": "$resy"})
            
            # Add default BSDF
            bsdf = ET.SubElement(root, "bsdf", {
                "type": "twosided", 
                "id": "default-bsdf", 
                "name": "default-bsdf"
            })
            ET.SubElement(bsdf, "bsdf", {"type": "diffuse", "name": "bsdf"})
            
            # Default emitter constant light radiance
            emitter = ET.SubElement(root, "emitter", {"type": "constant"})
            value = "0.6, 0.6, 0.6"
            radiance = ET.SubElement(emitter, "rgb", {"name": "radiance", "value": "0.6, 0.6, 0.6"})
            
            # Add shape if specified
            if shape_file:
                shape = ET.SubElement(root, "shape", {"type": "obj"})
                ET.SubElement(shape, "string", {
                    "name": "filename", 
                    "value": os.path.abspath(shape_file)
                })
                ET.SubElement(shape, "boolean", {"name": "face_normals", "value": "true"})
                ET.SubElement(shape, "ref", {"id": "default-bsdf", "name": "bsdf"})
            
            # Create the tree and write to file
            tree = ET.ElementTree(root)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write the XML with proper encoding and declaration
            tree.write(output_path, encoding="utf-8", xml_declaration=True)
            logger.debug(f"Created basic scene XML at {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating basic scene XML: {e}")
            raise
    
    @staticmethod
    @timeit(log_level="debug")
    def create_xml_from_scene_dict(scene_dict: Dict[str, Any], output_path: str) -> None:
        """
        Create an XML file from a scene dictionary.
        
        Args:
            scene_dict: Dictionary representing scene data
            output_path: Path to save the XML file
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Converting Python dict to XML
        root = ET.Element("scene")
        
        # Helper function to add elements recursively
        def add_elements(parent, data):
            if isinstance(data, dict):
                for key, value in data.items():
                    if key == "type":
                        parent.set("type", str(value))
                    else:
                        if isinstance(value, dict) and "type" in value:
                            sub_elem = ET.SubElement(parent, value["type"])
                            sub_elem.set("name", key)
                            add_elements(sub_elem, value)
                        elif isinstance(value, list):
                            sub_elem = ET.SubElement(parent, "list")
                            sub_elem.set("name", key)
                            for item in value:
                                add_elements(sub_elem, item)
                        else:
                            sub_elem = ET.SubElement(parent, "string" if isinstance(value, str) else "float")
                            sub_elem.set("name", key)
                            sub_elem.set("value", str(value))
            else:
                # Handle non-dict values
                parent.set("value", str(data))
        
        # Start the recursion with the scene dictionary
        add_elements(root, scene_dict)
        
        # Create the tree and write to file
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        logger.debug(f"Scene XML written to {output_path}")
        
    @staticmethod
    @timeit(log_level="debug")
    def update_template_xml_with_params(xml_path: str, spp: int = 256, width: int = 1920, height: int = 1080) -> str:
        """
        Create a copy of the template XML file with updated rendering parameters.
        
        Args:
            xml_path: Path to the template XML file
            spp: Samples per pixel
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Path to the updated XML file
        """
        if not os.path.exists(xml_path):
            logger.error(f"Template XML file not found: {xml_path}")
            raise FileNotFoundError(f"Template XML file not found: {xml_path}")
            
        try:
            # Parse the template XML
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Update SPP default value
            spp_elem = root.find("./default[@name='spp']")
            if spp_elem is not None:
                spp_elem.set("value", str(spp))
                logger.debug(f"Set SPP to {spp}")
            
            # Update resolution width default value
            resx_elem = root.find("./default[@name='resx']")
            if resx_elem is not None:
                resx_elem.set("value", str(width))
                logger.debug(f"Set width to {width}")
                
            # Update resolution height default value
            resy_elem = root.find("./default[@name='resy']")
            if resy_elem is not None:
                resy_elem.set("value", str(height))
                logger.debug(f"Set height to {height}")
            
            # Save to a temporary file
            output_dir = os.path.join(config.OUTPUT_FOLDER, config.SCENE_FOLDER)
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(output_dir, f"template_custom_{spp}spp_{width}x{height}.xml")
            tree.write(output_path, encoding="utf-8", xml_declaration=True)
            logger.debug(f"Created custom template XML with SPP={spp}, resolution={width}x{height}: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error updating template XML with custom parameters: {e}")
            # Fall back to the original template in case of errors
            return xml_path
