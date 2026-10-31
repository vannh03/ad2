import os
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
from ultralytics.data.utils import IMG_FORMATS
from ultralytics.utils import LOGGER, TORCH_VERSION
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.torch_utils import TORCH_2_4, select_device
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class VisualAISearch:
    """A semantic image search system that leverages OpenAI's CLIP for generating high-quality image and text embeddings
    and NumPy cosine similarity for fast similarity-based retrieval.

    This class aligns image and text embeddings in a shared semantic space, enabling users to search large collections
    of images using natural language queries with high accuracy and speed.

    Attributes:
        data (str): Directory containing images.
        device (str): Computation device, e.g., 'cpu' or 'cuda'.
        index_path (str): Path to the numpy file storing image embeddings.
        data_path_npy (str): Path to the numpy file storing image paths.
        data_dir (Path): Path object for the data directory.
        model: Loaded CLIP model.
        index (np.ndarray): L2-normalized image embeddings used for cosine similarity search.
        image_paths (list[str]): List of image file paths.

    Methods:
        extract_image_feature: Extract CLIP embedding from an image.
        extract_text_feature: Extract CLIP embedding from text.
        load_or_build_index: Load existing embeddings or build them from images.
        search: Perform semantic search for similar images.

    Examples:
        Initialize and search for images
        >>> searcher = VisualAISearch(data="path/to/images", device="cuda")
        >>> results = searcher.search("a cat sitting on a chair", k=10)
    """

    def __init__(self, **kwargs):
        """Initialize the VisualAISearch class with the embedding index and CLIP model."""
        assert TORCH_2_4, f'VisualAISearch requires torch>=2.4 (found torch=={TORCH_VERSION})'
        from ultralytics.nn.text_model import build_text_model
        self.index_path = 'embeddings.npy'
        self.data_path_npy = 'paths.npy'
        self.data_dir = Path(kwargs.get('data', 'images'))
        self.device = select_device(kwargs.get('device', 'cpu'))
        if not self.data_dir.exists():
            from ultralytics.utils import ASSETS_URL
            LOGGER.warning(f'{self.data_dir} not found. Downloading images.zip from {ASSETS_URL}/images.zip')
            from ultralytics.utils.downloads import safe_download
            safe_download(url=f'{ASSETS_URL}/images.zip', unzip=True, retry=3)
            self.data_dir = Path('images')
        self.model = build_text_model('clip:ViT-B/32', device=self.device)
        self.index = None
        self.image_paths = []
        self.load_or_build_index()

    def extract_image_feature(self, path):
        """Extract CLIP image embedding from the given image path."""
        return self.model.encode_image(Image.open(path)).detach().cpu().numpy()

    def extract_text_feature(self, text):
        """Extract CLIP text embedding from the given text query."""
        return self.model.encode_text(self.model.tokenize([text])).detach().cpu().numpy()

    @staticmethod
    def _normalize(x):
        """L2-normalize each row of `x` so inner products equal cosine similarity.

        Args:
            x (np.ndarray): Feature array of shape (N, D).

        Returns:
            (np.ndarray): Row-wise L2-normalized array with the same shape as the input.
        """
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    def load_or_build_index(self):
        """Load existing image embeddings or build them from the image directory.

        Checks if the embeddings and image paths exist on disk. If found, loads them directly. Otherwise, builds the
        index by extracting features from all images in the data directory, L2-normalizes them, and saves both the
        embeddings and image paths for future use.
        """
        if Path(self.index_path).exists() and Path(self.data_path_npy).exists():
            LOGGER.info('Loading existing embeddings...')
            self.index = np.load(self.index_path)
            self.image_paths = np.load(self.data_path_npy)
            return
        LOGGER.info('Building embeddings from images...')
        vectors = []
        for file in self.data_dir.iterdir():
            if file.suffix.lower().lstrip('.') not in IMG_FORMATS:
                continue
            try:
                vectors.append(self.extract_image_feature(file))
                self.image_paths.append(file.name)
            except Exception as e:
                LOGGER.warning(f'Skipping {file.name}: {e}')
        if not vectors:
            raise RuntimeError('No image embeddings could be generated.')
        vectors = np.vstack(vectors).astype('float32')
        self.index = self._normalize(vectors)
        np.save(self.index_path, self.index)
        np.save(self.data_path_npy, np.array(self.image_paths))
        LOGGER.info(f'Indexed {len(self.image_paths)} images.')

    def search(self, query, k=30, similarity_thresh=0.1):
        """Return top-k semantically similar images to the given query.

        Args:
            query (str): Natural language text query to search for.
            k (int, optional): Maximum number of results to return.
            similarity_thresh (float, optional): Minimum similarity threshold for filtering results.

        Returns:
            (list[str]): List of image filenames ranked by similarity score.

        Examples:
            Search for images matching a query
            >>> searcher = VisualAISearch(data="images")
            >>> results = searcher.search("red car", k=5, similarity_thresh=0.2)
        """
        text_feat = self._normalize(self.extract_text_feature(query).astype('float32'))
        scores = self.index @ text_feat[0]
        top_k = np.argsort(scores)[::-1][:max(k, 0)]
        results = [(self.image_paths[i], float(scores[i])) for i in top_k if scores[i] >= similarity_thresh]
        results.sort(key=lambda x: x[1], reverse=True)
        LOGGER.info('\nRanked Results:')
        for (name, score) in results:
            LOGGER.info(f'  - {name} | Similarity: {score:.4f}')
        return [r[0] for r in results]

    def __call__(self, query):
        """Direct call interface for the search function."""
        return self.search(query)

class SearchApp:
    """A Flask-based web interface for semantic image search with natural language queries.

    This class provides a clean, responsive frontend that enables users to input natural language queries and instantly
    view the most relevant images retrieved from the indexed database.

    Attributes:
        render_template: Flask template rendering function.
        request: Flask request object.
        searcher (VisualAISearch): Instance of the VisualAISearch class.
        app (Flask): Flask application instance.

    Methods:
        index: Process user queries and display search results.
        run: Start the Flask web application.

    Examples:
        Start a search application
        >>> app = SearchApp(data="path/to/images", device="cuda")
        >>> app.run(debug=True)
    """

    def __init__(self, data='images', device=None):
        """Initialize the SearchApp with VisualAISearch backend.

        Args:
            data (str, optional): Path to directory containing images to index and search.
            device (str, optional): Device to run inference on (e.g. 'cpu', 'cuda').
        """
        check_requirements('flask>=3.0.1')
        from flask import Flask, render_template, request
        self.render_template = render_template
        self.request = request
        self.searcher = VisualAISearch(data=data, device=device)
        self.app = Flask(__name__, template_folder='templates', static_folder=Path(data).resolve(), static_url_path='/images')
        self.app.add_url_rule('/', view_func=self.index, methods=['GET', 'POST'])

    def index(self):
        """Process user query and display search results in the web interface."""
        results = []
        if self.request.method == 'POST':
            query = self.request.form.get('query', '').strip()
            results = self.searcher(query)
        return self.render_template('similarity-search.html', results=results)

    def run(self, debug=False):
        """Start the Flask web application server."""
        self.app.run(debug=debug)