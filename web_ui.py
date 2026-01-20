from flask import Flask, render_template_string, request, jsonify
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import torch

app = Flask(__name__)

# Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "ecommerce_products"

# Initialize
print("="*60)
print("🚀 Initializing E-commerce Search UI")
print("="*60)

# Check CUDA
try:
    cuda_available = torch.cuda.is_available()
    device = 'cuda' if cuda_available else 'cpu'
    if cuda_available:
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  Running on CPU")
except:
    device = 'cpu'
    print("⚠️  Running on CPU")

# Load model
print(f"📦 Loading embedding model on {device.upper()}...")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
print("✅ Model loaded!")

# Connect to Qdrant
print(f"🔌 Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Verify collection exists
try:
    collection_info = client.get_collection(collection_name=COLLECTION_NAME)
    print(f"✅ Connected to collection '{COLLECTION_NAME}'")
    print(f"📊 Total products: {collection_info.points_count}")
except Exception as e:
    print(f"❌ Error: Collection '{COLLECTION_NAME}' not found!")
    print(f"   Make sure you ran embed.py first to index your data.")
    print(f"   Error details: {e}")

print("="*60)

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>E-commerce Semantic Search</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        .header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
            padding: 30px;
        }

        .header h1 {
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }

        .search-box {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            margin-bottom: 30px;
        }

        .search-input-wrapper {
            position: relative;
            margin-bottom: 20px;
        }

        .search-input {
            width: 100%;
            padding: 20px 60px 20px 20px;
            font-size: 1.1em;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            transition: all 0.3s;
        }

        .search-input:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        .search-btn {
            position: absolute;
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            transition: all 0.3s;
        }

        .search-btn:hover {
            transform: translateY(-50%) scale(1.05);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }

        .search-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }

        .stats {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }

        .stat-box {
            flex: 1;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 15px;
            border-radius: 12px;
            color: white;
            text-align: center;
        }

        .stat-box h3 {
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 5px;
        }

        .stat-box p {
            font-size: 1.5em;
            font-weight: bold;
        }

        .results {
            display: grid;
            gap: 20px;
        }

        .result-card {
            background: white;
            border-radius: 16px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: all 0.3s;
            animation: slideIn 0.4s ease-out;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .result-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        }

        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 2px solid #f0f0f0;
        }

        .result-rank {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }

        .result-score {
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 1em;
        }

        .result-details {
            display: grid;
            gap: 10px;
        }

        .detail-row {
            display: flex;
            padding: 8px 0;
        }

        .detail-label {
            font-weight: 600;
            color: #667eea;
            min-width: 150px;
            margin-right: 10px;
        }

        .detail-value {
            color: #333;
            flex: 1;
        }

        .loading {
            text-align: center;
            padding: 40px;
            color: white;
            font-size: 1.2em;
        }

        .no-results {
            text-align: center;
            padding: 60px;
            background: white;
            border-radius: 16px;
            color: #666;
        }

        .no-results h2 {
            margin-bottom: 10px;
            color: #333;
        }

        .error-box {
            background: #fee;
            border: 2px solid #fcc;
            color: #c33;
            padding: 20px;
            border-radius: 12px;
            margin: 20px 0;
        }

        .examples {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 12px;
            margin-top: 10px;
        }

        .examples h3 {
            color: white;
            margin-bottom: 10px;
            font-size: 1em;
        }

        .example-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }

        .example-chip {
            background: rgba(255,255,255,0.2);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s;
            border: 1px solid rgba(255,255,255,0.3);
        }

        .example-chip:hover {
            background: rgba(255,255,255,0.3);
            transform: scale(1.05);
        }

        .spinner {
            border: 4px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top: 4px solid white;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Semantic Product Search</h1>
            <p>Describe what you're looking for in natural language</p>
        </div>

        <div class="search-box">
            <div class="search-input-wrapper">
                <input 
                    type="text" 
                    class="search-input" 
                    id="searchInput" 
                    placeholder="E.g., 'comfortable running shoes for women' or 'wireless headphones with noise cancellation'"
                    autocomplete="off"
                >
                <button class="search-btn" id="searchBtn" onclick="search()">Search</button>
            </div>

            <div class="examples">
                <h3>💡 Try these examples:</h3>
                <div class="example-chips">
                    <span class="example-chip" onclick="setExample('laptop for gaming')">laptop for gaming</span>
                    <span class="example-chip" onclick="setExample('comfortable running shoes')">comfortable running shoes</span>
                    <span class="example-chip" onclick="setExample('wireless headphones')">wireless headphones</span>
                    <span class="example-chip" onclick="setExample('coffee maker')">coffee maker</span>
                </div>
            </div>
        </div>

        <div id="statsContainer" style="display: none;">
            <div class="stats">
                <div class="stat-box">
                    <h3>Results Found</h3>
                    <p id="resultCount">0</p>
                </div>
                <div class="stat-box">
                    <h3>Search Time</h3>
                    <p id="searchTime">0ms</p>
                </div>
                <div class="stat-box">
                    <h3>Best Match</h3>
                    <p id="bestMatch">0%</p>
                </div>
            </div>
        </div>

        <div id="results" class="results"></div>
    </div>

    <script>
        const searchInput = document.getElementById('searchInput');
        const searchBtn = document.getElementById('searchBtn');
        const resultsDiv = document.getElementById('results');
        const statsContainer = document.getElementById('statsContainer');

        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') search();
        });

        function setExample(text) {
            searchInput.value = text;
            search();
        }

        async function search() {
            const query = searchInput.value.trim();
            if (!query) return;

            searchBtn.disabled = true;
            searchBtn.textContent = 'Searching...';
            resultsDiv.innerHTML = '<div class="loading"><div class="spinner"></div><p>Searching for products...</p></div>';
            statsContainer.style.display = 'none';

            try {
                const startTime = performance.now();
                const response = await fetch('/search', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query, top_k: 10 })
                });

                const data = await response.json();
                const endTime = performance.now();
                const searchTime = Math.round(endTime - startTime);

                if (data.error) {
                    resultsDiv.innerHTML = `
                        <div class="error-box">
                            <h2>❌ Error</h2>
                            <p>${data.error}</p>
                        </div>
                    `;
                } else {
                    displayResults(data.results, searchTime);
                }
            } catch (error) {
                resultsDiv.innerHTML = `
                    <div class="error-box">
                        <h2>❌ Connection Error</h2>
                        <p>Could not connect to the search server. Make sure the server is running.</p>
                        <p>Error: ${error.message}</p>
                    </div>
                `;
            } finally {
                searchBtn.disabled = false;
                searchBtn.textContent = 'Search';
            }
        }

        function displayResults(results, searchTime) {
            if (!results || results.length === 0) {
                resultsDiv.innerHTML = `
                    <div class="no-results">
                        <h2>🔍 No Results Found</h2>
                        <p>Try a different search term or browse our examples above.</p>
                    </div>
                `;
                statsContainer.style.display = 'none';
                return;
            }

            document.getElementById('resultCount').textContent = results.length;
            document.getElementById('searchTime').textContent = searchTime + 'ms';
            document.getElementById('bestMatch').textContent = Math.round(results[0].score * 100) + '%';
            statsContainer.style.display = 'block';

            resultsDiv.innerHTML = results.map((result, index) => {
                const detailsHtml = Object.entries(result.payload)
                    .filter(([key]) => key !== 'searchable_text')
                    .map(([key, value]) => `
                        <div class="detail-row">
                            <div class="detail-label">${key}:</div>
                            <div class="detail-value">${value}</div>
                        </div>
                    `).join('');

                return `
                    <div class="result-card" style="animation-delay: ${index * 0.05}s">
                        <div class="result-header">
                            <div class="result-rank">#${index + 1}</div>
                            <div class="result-score">${Math.round(result.score * 100)}% Match</div>
                        </div>
                        <div class="result-details">
                            ${detailsHtml}
                        </div>
                    </div>
                `;
            }).join('');
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.json
        query = data.get('query', '')
        top_k = data.get('top_k', 10)

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        print(f"\n🔍 Searching for: '{query}'")

        # Generate query embedding
        query_embedding = model.encode(query, normalize_embeddings=True).tolist()

        # Use query_points method (available in your Qdrant version)
        search_results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=top_k
        )
        
        # Format results
        results = [{
            'id': point.id,
            'score': point.score,
            'payload': point.payload
        } for point in search_results.points]
        
        print(f"✅ Found {len(results)} results")
        return jsonify({'results': results})

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌐 Starting Web UI Server")
    print("="*60)
    print("\n📱 Open your browser: http://localhost:5000")
    print("⌨️  Press CTRL+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)