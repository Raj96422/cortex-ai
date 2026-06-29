"""
Unit Tests for Cortex AI Frontend Modules.
Verifies that all Streamlit pages, styles, and UI component functions load and compile 
without syntax or import errors.
"""

import sys
import unittest
from pathlib import Path

# Ensure workspace root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestFrontendLoading(unittest.TestCase):
    """Test suite ensuring frontend pages and component modules compile successfully."""

    def test_components_module_loading(self):
        """Verifies that the custom ui_components module is importable."""
        from ui.components.ui_components import (
            get_pipeline,
            inject_custom_css,
            render_header,
            render_footer,
            render_metric_card,
            render_citation_card,
        )
        self.assertTrue(callable(inject_custom_css))
        self.assertTrue(callable(render_header))
        self.assertTrue(callable(render_footer))
        self.assertTrue(callable(render_metric_card))
        self.assertTrue(callable(render_citation_card))
        self.assertTrue(callable(get_pipeline))

    def test_main_app_compiles(self):
        """Verifies that ui/app.py does not contain any syntax or structural compile errors."""
        app_path = Path(__file__).resolve().parent.parent / "ui" / "app.py"
        self.assertTrue(app_path.exists())
        
        with open(app_path, "r", encoding="utf-8") as f:
            source_code = f.read()
            
        # Compile source to check syntax
        compiled = compile(source_code, str(app_path), "exec")
        self.assertIsNotNone(compiled)

    def test_pages_compile(self):
        """Verifies that all pages in the ui/pages/ directory compile successfully."""
        pages_dir = Path(__file__).resolve().parent.parent / "ui" / "pages"
        self.assertTrue(pages_dir.exists())

        for page_file in pages_dir.glob("*.py"):
            with open(page_file, "r", encoding="utf-8") as f:
                source_code = f.read()
            # Compile source to check syntax
            compiled = compile(source_code, str(page_file), "exec")
            self.assertIsNotNone(compiled, f"Failed to compile page: {page_file.name}")


if __name__ == "__main__":
    unittest.main()
