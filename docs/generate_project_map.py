#!/usr/bin/env python3
"""
generate_project_map.py
Automatically generates a comprehensive project_map.md file with dynamic tree structure and module documentation.
"""

import os
import re
import ast
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class ProjectMapGenerator:
    """Generates a project map documentation with dynamic tree structure and module docstrings."""
    
    def __init__(self, project_root: str = "."):
        """Initialize the generator with project root directory."""
        self.project_root = Path(project_root)
        self.ignore_dirs = {
            "__pycache__", 
            ".git", 
            ".pytest_cache", 
            "node_modules",
            ".ipynb_checkpoints",
            "assets",
            "variables",
            "debug"
        }
        self.ignore_files = {
            "__init__.py",
            ".pyc",
            ".pyo", 
            ".DS_Store"
        }

    def extract_docstring(self, file_path: Path) -> Optional[str]:
        """Extract the module docstring from a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            docstring = ast.get_docstring(tree)
            return docstring.strip() if docstring else None
        except Exception:
            return None

    def collect_python_docstrings(self) -> Dict[str, str]:
        """Collect docstrings from all Python files in the project."""
        docstrings = {}
        
        # Define directories to scan for Python files
        target_dirs = [
            "code/src",
            "code/utils", 
            "code/ux_ui",
            "tests"
        ]
        
        for target_dir in target_dirs:
            dir_path = self.project_root / target_dir
            if not dir_path.exists():
                continue
                
            # Find all Python files in this directory
            for py_file in dir_path.rglob("*.py"):
                # Skip ignored files and directories
                if any(ignore in str(py_file) for ignore in self.ignore_dirs):
                    continue
                if py_file.name in self.ignore_files:
                    continue
                
                # Extract docstring
                docstring = self.extract_docstring(py_file)
                if docstring:
                    # Create relative path for display
                    relative_path = py_file.relative_to(self.project_root)
                    docstrings[str(relative_path)] = docstring
        
        return docstrings

    def generate_docstring_section(self) -> str:
        """Generate the module documentation section with docstrings."""
        docstrings = self.collect_python_docstrings()
        
        if not docstrings:
            return ""
        
        # Sort by path for consistent ordering
        sorted_items = sorted(docstrings.items())
        
        lines = ["## Module Documentation", ""]
        
        for file_path, docstring in sorted_items:
            lines.append(f"### `{file_path}`")
            lines.append("")
            lines.append(docstring)
            lines.append("")
        
        return "\n".join(lines)

    def generate_tree_structure(self) -> str:
        """Generate a concise tree-like structure based on actual project directory scanning."""
        tree_lines = ["workspace/"]
        
        def scan_directory(path: Path, prefix: str = "", max_depth: int = 3, current_depth: int = 0) -> None:
            if current_depth >= max_depth:
                return
            
            try:
                items = []
                for item in path.iterdir():
                    # Skip hidden files and ignored directories
                    if item.name.startswith('.') and item.name not in ['.devcontainer']:
                        continue
                    if item.name in self.ignore_dirs:
                        continue
                    
                    # Only show directories
                    if item.is_dir():
                        items.append(item)
                
                # Sort directories
                items.sort(key=lambda x: x.name.lower())
                
                for i, item in enumerate(items):
                    is_last = i == len(items) - 1
                    current_prefix = "└── " if is_last else "├── "
                    tree_lines.append(f"{prefix}{current_prefix}{item.name}")
                    
                    # Add description for key directories
                    description = self._get_directory_description(item)
                    if description and current_depth < 2:
                        tree_lines[-1] += f"   → {description}"
                    
                    # Recurse into subdirectories
                    if current_depth < max_depth - 1:
                        # For data directories, show summary instead of contents
                        if item.name in ['v1.0_raw_data', 'v2.0_zip_files', 'v3.0_las_files']:
                            subdir_count = len([x for x in item.iterdir() if x.is_dir()])
                            next_prefix = prefix + ("    " if is_last else "│   ")
                            tree_lines.append(f"{next_prefix}├── [{subdir_count} field directories]")
                        elif item.name == 'zip_files':
                            file_count = len([x for x in item.iterdir() if x.is_file()])
                            next_prefix = prefix + ("    " if is_last else "│   ")
                            tree_lines.append(f"{next_prefix}├── [{file_count} zip archives]")
                        else:
                            next_prefix = prefix + ("    " if is_last else "│   ")
                            scan_directory(item, next_prefix, max_depth, current_depth + 1)
                            
            except PermissionError:
                pass
        
        scan_directory(self.project_root, max_depth=3)
        return "\n".join(tree_lines)
    
    def _get_directory_description(self, path: Path) -> str:
        """Get a brief description for key directories."""
        descriptions = {
            '.devcontainer': 'Development container configurations',
            'code': 'Main source code and utilities',
            'src': 'Core application modules',
            'data_preprocessing': 'Feature engineering and normalization',
            'neural_network': 'ML models and training pipeline',
            'utils': 'Helper utilities and tools',
            'ux_ui': 'Interactive UI components',
            'data': 'Raw and processed datasets',
            'v1.0_raw_data': 'Original KGS data files',
            'v2.0_zip_files': 'Compressed data archives',
            'v3.0_las_files': 'Processed LAS files',
            'debug': 'Debug and experimental scripts',
            'docs': 'Project documentation',
            'tests': 'Automated test suite',
            'reports': 'Analysis reports and outputs',
            'plots': 'Generated visualizations'
        }
        return descriptions.get(path.name, "")

    def generate_project_map(self) -> str:
        """Generate the complete project map documentation."""
        print("🔍 Analyzing project structure...")
        
        print("🌳 Generating directory tree...")
        tree_structure = self.generate_tree_structure()
        
        print("📚 Collecting module docstrings...")
        docstring_section = self.generate_docstring_section()
        
        # Assemble the complete document with only dynamic content
        project_map = f"""# Project Map

*Auto-generated documentation - Last updated: {Path(__file__).stat().st_mtime}*

## Directory Structure

```
{tree_structure}
```

{docstring_section}

---

*This document is automatically generated by `generate_project_map.py`. To update, run the script from the project root directory.*
"""
        
        return project_map

    def save_project_map(self, output_path: str = "docs/project_map.mdc"):
        """Generate and save the project map to file."""
        project_map_content = self.generate_project_map()
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the content
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(project_map_content)
        
        print(f"✅ Project map generated successfully: {output_file}")
        return output_file


if __name__ == "__main__":
    # Always use the workspace root as project root
    if Path(__file__).parent.name == "docs":
        # Script is in docs folder, go up one level
        project_root = Path(__file__).parent.parent
    else:
        # Script is in root folder
        project_root = Path(__file__).parent
        
    generator = ProjectMapGenerator(project_root)
    
    print("🚀 Generating complete project map...")
    generator.save_project_map() 