import asyncio
import os
from playwright.async_api import async_playwright

async def capture_elements():
    async with async_playwright() as p:
        # Browser setup with high device scale factor for high resolution
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1200, 'height': 800},
            device_scale_factor=2.0  # 2x scale for high resolution
        )
        page = await context.new_page()

        # Load the local HTML file
        file_path = r'c:\Users\haide\Desktop\fyp2\elsagate_research_summary.html'
        file_url = f'file:///{file_path.replace("\\", "/")}'
        print(f"Loading {file_url}...")
        
        await page.goto(file_url, wait_until="networkidle")

        # Create output directory
        output_dir = r'c:\Users\haide\Desktop\fyp2\extracted_images'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")

        # 1. Capture Tables
        tables = await page.query_selector_all("table")
        for i, table in enumerate(tables):
            # Try to find a heading before the table for naming
            heading = await page.evaluate_handle(
                "(node) => { let prev = node.previousElementSibling; while(prev && !['H2', 'H3'].includes(prev.tagName)) { prev = prev.previousElementSibling; } return prev ? prev.innerText : null; }",
                table
            )
            name = await heading.json_value()
            name = name.lower().replace(":", "").replace(" ", "_") if name else f"table_{i+1}"
            save_path = os.path.join(output_dir, f"{name}.png")
            
            # Add some padding for the screenshot
            await table.screenshot(path=save_path)
            print(f"Saved table: {save_path}")

        # 2. Capture specific visualization blocks or images
        # We target .visualization-block and .nasnet-grid or just all imgs
        images = await page.query_selector_all("img")
        for i, img in enumerate(images):
            # Get alt text for naming
            alt = await img.get_attribute("alt")
            name = alt.lower().replace(" ", "_") if alt else f"figure_{i+1}"
            save_path = os.path.join(output_dir, f"{name}.png")
            
            await img.screenshot(path=save_path)
            print(f"Saved image: {save_path}")

        # 3. Capture specific custom blocks (like the dataset bars)
        # Find divs with class "dataset-bar" and capture their parent container for context
        dataset_sections = await page.query_selector_all("section div:has(.dataset-bar)")
        for i, section in enumerate(dataset_sections):
            save_path = os.path.join(output_dir, f"dataset_profile_{i+1}.png")
            await section.screenshot(path=save_path)
            print(f"Saved dataset section: {save_path}")

        await browser.close()
        print("\nAll captures complete!")

if __name__ == "__main__":
    asyncio.run(capture_elements())
