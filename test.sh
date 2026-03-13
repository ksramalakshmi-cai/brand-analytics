# 1. Register logo
curl -s -X POST "http://localhost:8000/logos" \
  -F "logo_id=coke" \
  -F "name=Coca-cola" \
  -F "detection_method=ocr" \
  -F "images=@/home/sagemaker-user/ramalakshmi/brand-analytics/images.png"

# 2. Edit logo
curl -s -X PATCH "http://localhost:8000/logos/coke" \
  -F "name=Coca-cola" \
  -F "detection_method=both" \
  -F "images=@/path/to/extra.png"

3. Delete logo
curl -s -X DELETE "http://localhost:8000/logos/coke"

# 4. Send logo images (register with images)
curl -s -X POST "http://localhost:8000/logos" \
  -F "logo_id=test_brand" \
  -F "name=Test Brand" \
  -F "images=@./logo1.png" \
  -F "images=@./logo2.jpg"

# 5. Send video for processing
curl -s -X POST "http://localhost:8000/process" \
  -H "Content-Type: application/json" \
  -d '{"url":"s3://brand-analytics-assets/videos/videoplayback-2.mp4","video_id":"post_123","engagements":100,"target_logos":["aramco", "dp_world", "emirates", "google", "hyundai", "marriot_bonvoy", "royal_stag", "sobha_realty"]}'

# 6. Get video processing update (replace JOB_ID with value from step 5)
curl -s "http://localhost:8000/jobs/JOB_ID"

# 7. Get metrics – all logos
curl -s "http://localhost:8000/stats"

# 7b. Get metrics – specific logos
curl -s "http://localhost:8000/stats?logo_ids=emirates,aramco"