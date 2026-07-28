import asyncio
import os
from playwright.async_api import async_playwright

async def capture_wbs():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1200, 'height': 1200},
            device_scale_factor=2.0
        )
        page = await context.new_page()

        file_path = r'c:\Users\haide\Desktop\fyp2\wbs_chart.html'
        file_url = f'file:///{file_path.replace("\\", "/")}'
        await page.goto(file_url, wait_until="networkidle")

        output_path = r'c:\Users\haide\Desktop\fyp2\extracted_images\wbs_chart.png'
        
        # Capture the main container
        container = await page.query_selector("#wbs-chart")
        await container.screenshot(path=output_path)
        print(f"Saved WBS chart: {output_path}")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(capture_wbs())
