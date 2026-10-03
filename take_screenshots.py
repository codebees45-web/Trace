import asyncio
from playwright.async_api import async_playwright
import os

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # Using a large viewport to make sure everything fits nicely
        page = await browser.new_page(viewport={"width": 1440, "height": 900})
        
        url = 'http://localhost:5173'
        await page.goto(url)
        await page.wait_for_timeout(3000) # wait for animation/hero

        # 1. TRACE Landing Page Interface
        await page.screenshot(path='TRACE_Landing_Page_Interface.png', full_page=True)
        print("Captured TRACE Landing Page Interface")

        # 2. Image Upload and Recognition Interface
        await page.click('button:has-text("Upload")')
        await page.wait_for_timeout(500)
        await page.screenshot(path='Image_Upload_and_Recognition_Interface.png')
        print("Captured Image Upload Interface")

        # 3, 4, 5, 6 from Showcase gallery
        await page.click('button:has-text("Showcase gallery")')
        await page.wait_for_selector('.results-panel', timeout=15000)
        await page.wait_for_timeout(2000)

        # Let's take a screenshot of the first result card
        demo_card = await page.query_selector('.demo-card')
        if demo_card:
            # 3. Recognition Result and Confidence View
            # 4. Top-K Identity Candidate View
            # 5. Face Quality Assessment View
            # 6. Masked Input and Reconstructed Output View
            await demo_card.screenshot(path='Recognition_Result_and_Confidence_View.png')
            print("Captured Recognition Result and Confidence View")
            
            topk = await demo_card.query_selector('.topk-list')
            if topk:
                await topk.screenshot(path='Top_K_Identity_Candidate_View.png')
                print("Captured Top-K Identity Candidate View")
                
            badge = await demo_card.query_selector('.quality-badge')
            if badge:
                await badge.screenshot(path='Face_Quality_Assessment_View.png')
                print("Captured Face Quality Assessment View")
                
            slider = await demo_card.query_selector('.slider-container')
            if slider:
                await slider.screenshot(path='Masked_Input_and_Reconstructed_Output_View.png')
                print("Captured Masked Input and Reconstructed Output View")

        # 8. Batch Prediction Interface
        await page.click('button:has-text("Batch upload")')
        await page.wait_for_timeout(1000)
        await page.screenshot(path='Batch_Prediction_Interface.png')
        print("Captured Batch Prediction Interface")

        # 9. Multi-Face Detection Interface
        await page.click('button:has-text("⚡ Live Detect")')
        await page.wait_for_timeout(2000)
        await page.screenshot(path='Multi_Face_Detection_Interface.png')
        print("Captured Multi-Face Detection Interface")

        # 7. Recognition History View
        # Log in first
        await page.click('button.nav-ghost-btn:has-text("Log in")')
        await page.wait_for_selector('.modal-card')
        
        # Click Sign up tab
        await page.click('button:has-text("Sign up")')
        await page.fill('input[type="email"]', 'admin@example.com')
        await page.fill('input[type="password"]', 'password1234')
        await page.click('button.btn-primary:has-text("Create account")')
        await page.wait_for_timeout(2000)
        
        # Click history button (user email)
        await page.click('button.nav-user')
        await page.wait_for_timeout(1500)
        await page.screenshot(path='Recognition_History_View.png')
        print("Captured Recognition History View")

        await browser.close()

asyncio.run(main())
