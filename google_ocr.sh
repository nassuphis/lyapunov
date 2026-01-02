KEY="AIzaSyADIWQ4H9D4D7mQ47RDEq70VDkFD1hmG5w"
IMG_DATA=$(base64 -i ocr_nn14loc101_00081.png | tr -d '\n')

curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=$KEY" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d "{
      \"contents\": [{
        \"parts\":[
          {\"text\": \"Transcribe the text in this image. Output only the raw text as a single line. This is a technical configuration string.\"},
          {\"inline_data\": {
            \"mime_type\":\"image/png\",
            \"data\": \"$IMG_DATA\"
          }}
        ]
      }],
      \"safetySettings\": [
        { \"category\": \"HARM_CATEGORY_HARASSMENT\", \"threshold\": \"BLOCK_NONE\" },
        { \"category\": \"HARM_CATEGORY_HATE_SPEECH\", \"threshold\": \"BLOCK_NONE\" },
        { \"category\": \"HARM_CATEGORY_SEXUALLY_EXPLICIT\", \"threshold\": \"BLOCK_NONE\" },
        { \"category\": \"HARM_CATEGORY_DANGEROUS_CONTENT\", \"threshold\": \"BLOCK_NONE\" }
      ]
    }" | jq -r '.candidates[0].content.parts[0].text'

