import os
from flask import Flask, request, jsonify
import google.generativeai as genai
from tavily import TavilyClient

app = Flask(__name__)

# Configure your API Keys (Set these in your server environment variables)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
tavily = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

@app.route('/analyze', methods=['POST'])
def analyze_product():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image_file = request.files['image']
    
    # 1. Gemini identifies the product
    model = genai.GenerativeModel('gemini-1.5-flash')
    vision_response = model.generate_content(["Identify this product and give me its exact name.", image_file.read()])
    product_name = vision_response.text.strip()

    # 2. Tavily finds the best prices
    search_query = f"buy {product_name} best price online"
    search_results = tavily.search(query=search_query, max_results=3)

    return jsonify({
        "product": product_name,
        "deals": search_results['results']
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
