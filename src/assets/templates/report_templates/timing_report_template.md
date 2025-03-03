# Rendering Timing Report

## Overview

- **Scene Name:** {{scene_name}}
- **Timestamp:** {{timestamp}}
- **Total Duration:** {{total_duration}}
- **Total Frames:** {{total_frames}}
- **Total Views:** {{total_views}}
- **Quality Preset:** {{quality}}

## Performance Summary

- **Average Time per Frame:** {{avg_frame_time}}
- **Fastest Frame:** Frame {{fastest_frame.index}} ({{fastest_frame.time}})
- **Slowest Frame:** Frame {{slowest_frame.index}} ({{slowest_frame.time}})
- **Average Time per View:** {{avg_view_time}}

## Cache Statistics

- **Cache Hits:** {{cache_hits}} ({{cache_hit_rate}}%)
- **Cache Misses:** {{cache_misses}}
- **Total Cache Operations:** {{total_cache_operations}}

## Frame Breakdown

| Frame | Duration | Views | Avg View Time | Status |
|-------|----------|-------|--------------|--------|
{{#each frames}}
| {{index}} | {{duration}} | {{view_count}} | {{avg_view_duration}} | {{status}} |
{{/each}}

## View Breakdown

| View | Total Time | Avg Time | Frames | Cache Hits |
|------|------------|----------|--------|------------|
{{#each views}}
| {{name}} | {{total_time}} | {{avg_time}} | {{frame_count}} | {{cache_hits}} |
{{/each}}

## System Information

- **Device:** {{device}}
- **CPU Cores:** {{cpu_cores}}
- **Memory:** {{memory}}
- **Mitsuba Variant:** {{mitsuba_variant}}
- **Python Version:** {{python_version}}
