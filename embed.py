import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm
import os

class EcommerceSemanticSearch:
    def __init__(self, qdrant_host="localhost", qdrant_port=6333, collection_name="ecommerce_products", use_gpu=True):
        """Initialize the semantic search system with Qdrant."""
        print("🚀 Initializing E-commerce Semantic Search System...")
        
        # Initialize Qdrant client
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        
        # Initialize embedding model - using best GPU-optimized model
        print("📦 Loading embedding model...")
        
        # Check if CUDA is actually available
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if use_gpu and not cuda_available:
                print("⚠️  CUDA not available, falling back to CPU mode")
                use_gpu = False
            elif use_gpu and cuda_available:
                print(f"✅ CUDA detected: {torch.cuda.get_device_name(0)}")
        except:
            use_gpu = False
            print("⚠️  PyTorch not found or CUDA unavailable, using CPU mode")
        
        device = 'cuda' if use_gpu else 'cpu'
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        print(f"✅ Model loaded on {device.upper()}!")
        print(f"📊 Embedding dimension: {self.embedding_dim}")
    
    def create_collection(self):
        """Create a new Qdrant collection for products."""
        print(f"🗄️  Creating collection '{self.collection_name}'...")
        
        # Delete collection if it exists
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            print("   Deleted existing collection")
        except:
            pass
        
        # Create new collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
        )
        print("✅ Collection created successfully!")
    
    def prepare_searchable_text(self, row):
        """Convert a product row into searchable text."""
        # Combine all columns into a meaningful text representation
        parts = []
        for col, value in row.items():
            if pd.notna(value) and str(value).strip():
                parts.append(f"{col}: {value}")
        return ". ".join(parts)
    
    def index_products(self, csv_file_path, batch_size=512):
        """Read CSV and index all products into Qdrant."""
        print(f"📂 Reading CSV file: {csv_file_path}")
        
        # Try reading CSV with different parameters to handle various formats
        try:
            df = pd.read_csv(csv_file_path)
        except pd.errors.ParserError:
            print("⚠️  Standard CSV parsing failed, trying with different settings...")
            try:
                # Try with different delimiter or error handling
                df = pd.read_csv(csv_file_path, on_bad_lines='skip', engine='python')
            except:
                # Try with explicit encoding
                df = pd.read_csv(csv_file_path, encoding='latin-1', on_bad_lines='skip')
        except UnicodeDecodeError:
            print("⚠️  Encoding issue detected, trying with latin-1 encoding...")
            df = pd.read_csv(csv_file_path, encoding='latin-1')
        
        print(f"📊 Found {len(df)} products")
        print(f"📋 Columns: {', '.join(df.columns.tolist())}")
        
        # Create collection
        self.create_collection()
        
        # Prepare all searchable texts first
        print("🔄 Preparing product texts...")
        searchable_texts = []
        for idx, row in df.iterrows():
            searchable_texts.append(self.prepare_searchable_text(row))
        
        # Generate embeddings in large batches on GPU (MUCH faster!)
        print(f"🚀 Generating embeddings on GPU (batch size: {batch_size})...")
        all_embeddings = self.model.encode(
            searchable_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Pre-normalize for faster cosine similarity
        )
        
        # Upload to Qdrant in batches
        print(f"📤 Uploading to Qdrant...")
        points = []
        upload_batch_size = 1000  # Larger batches for Qdrant upload
        
        for idx, (row, embedding) in enumerate(tqdm(zip(df.iterrows(), all_embeddings), 
                                                      total=len(df), 
                                                      desc="Uploading to Qdrant")):
            _, row_data = row
            
            # Create point with payload containing all product data
            point = PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload={
                    "searchable_text": searchable_texts[idx],
                    **{col: str(val) for col, val in row_data.items() if pd.notna(val)}
                }
            )
            points.append(point)
            
            # Upload batch when ready
            if len(points) >= upload_batch_size:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                points = []
        
        # Upload remaining points
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
        
        print(f"✅ Successfully indexed {len(df)} products!")
        
        # Show collection info
        collection_info = self.client.get_collection(collection_name=self.collection_name)
        print(f"📊 Collection stats: {collection_info.points_count} points")
    
    def search(self, query, top_k=10):
        """Search for products using natural language query."""
        print(f"🔍 Searching for: '{query}'")
        
        # Generate query embedding
        query_embedding = self.model.encode(query).tolist()
        
        # Search in Qdrant
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        return search_results
    
    def display_results(self, results):
        """Display search results in a readable format."""
        if not results:
            print("❌ No results found")
            return
        
        print(f"\n{'='*80}")
        print(f"🎯 Top {len(results)} Results:")
        print(f"{'='*80}\n")
        
        for i, result in enumerate(results, 1):
            print(f"#{i} - Score: {result.score:.4f} (ID: {result.id})")
            print("-" * 80)
            
            # Display payload (product details)
            for key, value in result.payload.items():
                if key != "searchable_text":  # Skip the combined text
                    print(f"  {key}: {value}")
            
            print(f"\n  [Searchable Text Preview]: {result.payload.get('searchable_text', '')[:200]}...")
            print("=" * 80 + "\n")


def main():
    """Main function to demonstrate the system."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║     E-COMMERCE SEMANTIC SEARCH WITH QDRANT                  ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize the search system with GPU acceleration
    # Set use_gpu=False if having CUDA issues
    search_system = EcommerceSemanticSearch(
        qdrant_host="localhost",
        qdrant_port=6333,
        collection_name="ecommerce_products",
        use_gpu=True  # Change to False for CPU-only mode
    )
    
    # Ask user for CSV file path
    csv_path = input("📁 Enter the path to your CSV file: ").strip()
    
    # Remove quotes if user copied path with quotes
    csv_path = csv_path.strip('"').strip("'")
    
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return
    
    # Check if it's actually a CSV file
    if not csv_path.lower().endswith('.csv'):
        print(f"⚠️  Warning: File doesn't have .csv extension. Is this correct?")
        confirm = input("Continue anyway? (y/n): ").strip().lower()
        if confirm != 'y':
            return
    
    # Index products with optimized batch size for GPU
    search_system.index_products(csv_path, batch_size=512)
    
    # Interactive search loop
    print("\n" + "="*80)
    print("🎉 Indexing complete! You can now search for products.")
    print("="*80 + "\n")
    
    while True:
        print("\n" + "-"*80)
        query = input("🔍 Enter your search query (or 'quit' to exit): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not query:
            continue
        
        # Perform search
        results = search_system.search(query, top_k=5)
        search_system.display_results(results)


if __name__ == "__main__":
    # OPTIMIZED FOR HIGH-END GPU (RTX 3500 Ada, 12GB VRAM)
    # 
    # Setup:
    # 1. Make sure Qdrant is running in Docker:
    #    docker run -p 6333:6333 qdrant/qdrant
    # 
    # 2. Install dependencies with CUDA support:
    #    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    #    pip install qdrant-client sentence-transformers pandas numpy tqdm
    # 
    # 3. Run this script:
    #    python ecommerce_search.py
    # 
    # Performance with your specs:
    # - RTX 3500 Ada (12GB VRAM) can process ~5000-10000 embeddings/second
    # - 26k products should index in under 10 seconds!
    # - Batch size 512 optimized for your VRAM
    # 
    # For even better quality (but slower), change model to:
    # 'all-mpnet-base-v2' (768 dim) - 2-3x slower but better accuracy
    
    main()