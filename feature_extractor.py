import cv2 as cv
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from skimage.measure import shannon_entropy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from joblib import Parallel, delayed
from itertools import islice
import warnings

# Default Configuration
CONFIG = {
    'sampling': {
        'strategy': 'uniform',  # Options: 'uniform', 'dense', 'random'
        'n_frames': 16,         # For uniform/random
        'frame_skip': 5,        # For dense
    },
    'resize_dim': (224, 224),
    'preprocess': {
        'denoise': True,
        'normalize_pixel': True, # Pixel value scaling 0-1
    },
    'normalization': 'minmax', # Options: 'minmax', 'standard'
    'n_jobs': -1,

    'lbp_radius': 3,
    'lbp_points': 8,

    'gabor': {
        'ksize': 31, # Increased for better texture capture
        'sigma': 4.0,
        'theta': 0,
        'lamda': 10.0,
        'gamma': 0.5,
        'phi': 0
    },

    'contour': {
        'count' : 3,
    },

    'lucas_kanade': {
        'max_corners': 20,
        'quality_level': 0.01,
        'min_distance': 10,
        'block_size': 7
    },
}

class TemporalFeatureExtractor:
    def __init__(self, window_len):
        self.window_len = window_len
        self.feature_buffer = []

    def update(self, frame_features: dict):
        self.feature_buffer.append(frame_features)
        if len(self.feature_buffer) > self.window_len:
            self.feature_buffer.pop(0)
        return self._get_temporal_features()

    def _get_temporal_features(self):
        features = {}
        if not self.feature_buffer:
            return features
            
        df = pd.DataFrame(self.feature_buffer)
        
        # Only compute temporal stats for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            features[f"{col}_temp_mean"] = df[col].mean()
            features[f"{col}_temp_std"] = df[col].std(ddof=0)
            
            if len(self.feature_buffer) > 1:
                deltas = df[col].diff()
                features[f"{col}_delta_mean"] = deltas.mean()
            else:
                features[f"{col}_delta_mean"] = 0.0
                
        # Simple motion trend
        if len(self.feature_buffer) > 2 and 'motion_avg_intensity' in df.columns:
            try:
                # Use range as x, intensity as y
                slope = np.polyfit(range(len(self.feature_buffer)), df['motion_avg_intensity'], 1)[0]
            except:
                slope = 0.0
        else:
            slope = 0.0
        features[f"motion_slope"] = slope

        return features

class VideoExtractorFeature:
    def __init__(self, config=None):
        self.config = config if config else CONFIG
        
        if self.config.get('normalization') == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
            
        # Initialize Gabor Kernel
        # Ensure parameters are integers/floats as expected by OpenCV
        g_params = self.config['gabor']
        self.gabor_kernel = cv.getGaborKernel(
            (int(g_params['ksize']), int(g_params['ksize'])),
            float(g_params['sigma']),
            float(g_params['theta']),
            float(g_params['lamda']),
            float(g_params['gamma']),
            float(g_params['phi']),
            ktype=cv.CV_32F
        )

    def _preprocess_frame(self, frame):
        """Apply denoising, resizing and normalization."""
        if frame is None:
            return None
            
        # Denoise
        if self.config['preprocess'].get('denoise'):
            frame = cv.GaussianBlur(frame, (5, 5), 0)
        
        # Resize
        frame = cv.resize(frame, self.config['resize_dim'], interpolation=cv.INTER_AREA)
        
        return frame

    def _get_frame_color_features(self, frame):
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        features = {}
        # RGB Histograms
        for i, colour in enumerate(['red', 'blue', 'green']):
            channel = rgb_frame[:, :, i]
            hist, _ = np.histogram(channel.ravel(), bins=10, range=(0, 256))
            hist = hist.astype('float')
            hist /= (hist.sum() + 1e-7)
            for j in range(len(hist)):
                features[f'color_{colour}_{j}'] = hist[j]

        # HSV Stats
        for i, column_name in enumerate(['h', 's', 'v']):
            channel = hsv_frame[:, :, i]
            mean = np.mean(channel)
            std = np.std(channel)
            
            features[f'moments_mean_{column_name}'] = mean
            features[f'moments_std_{column_name}'] = std
            
            if std > 1e-6:
                skew_val = skew(channel.flatten())
                features[f'moments_skew_{column_name}'] = 0 if np.isnan(skew_val) else skew_val
            else:
                features[f'moments_skew_{column_name}'] = 0

        avg_rgb  = np.mean(rgb_frame, axis=(0, 1))
        features['avg_color_r'] = avg_rgb[0]
        features['avg_color_g'] = avg_rgb[1]
        features['avg_color_b'] = avg_rgb[2]
        return features

    def _get_frame_glcm_features(self, grey_frame):
        features = {}
        # Using fewer distances/angles for efficiency while capturing texture
        distances = [1, 3]
        angles = [0, np.pi/2] # Horizontal and Vertical
        
        # GLCM requires integer types
        grey_frame_int = (grey_frame).astype(np.uint8)
        
        glcm = graycomatrix(grey_frame_int, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
        
        props = ['contrast', 'dissimilarity', 'homogeneity', 'correlation', 'energy']
        for prop in props:
            val = graycoprops(glcm, prop).ravel()
            # Average over all distances/angles to reduce feature dimensionality
            features[f'glcm_{prop}_mean'] = np.mean(val)
            features[f'glcm_{prop}_std'] = np.std(val)
            
        features['glcm_entropy'] = shannon_entropy(grey_frame)
        return features

    def _lbp_features(self, grey_frame):
        # LBP usually on integer images? scikit-image handles float but warns.
        # Ensure it works.
        lbp = local_binary_pattern(grey_frame, self.config['lbp_points'], self.config['lbp_radius'], method='uniform')
        # Uniform LBP histogram
        n_bins = self.config['lbp_points'] + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype('float')
        hist /= (hist.sum() + 1e-7)

        features = {}
        for i in range(len(hist)):
            features[f'lbp_{i}'] = hist[i]
        return features

    def _get_gabor_features(self, grey_frame):
        gabor_features = cv.filter2D(grey_frame, cv.CV_32F, self.gabor_kernel)
        
        mean = np.mean(gabor_features)
        std = np.std(gabor_features)
        features = {
            'gabor_mean': mean,
            'gabor_std': std
        }
        return features

    def _get_canny_features(self, grey_frame):
        sigma = 0.33
        v = np.median(grey_frame)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edges = cv.Canny(grey_frame, lower, upper)

        # Edge density
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        features = {'canny_edge_density': edge_density}
        return features

    def _get_contour_features(self, grey_frame):
        # Binary threshold
        _, img_th = cv.threshold(grey_frame, 127, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(img_th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        features = {}
        count = self.config['contour']['count']
        
        # Sort by area
        sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)
        
        for i in range(count):
            if i < len(sorted_contours):
                c = sorted_contours[i]
                area = cv.contourArea(c)
                perimeter = cv.arcLength(c, True)
                if perimeter == 0: perimeter = 1e-7
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                
                features[f'contour_{i}_area'] = area
                features[f'contour_{i}_circularity'] = circularity
            else:
                features[f'contour_{i}_area'] = 0
                features[f'contour_{i}_circularity'] = 0
        return features

    def _get_hog_features(self, grey_frame):
        # Using smaller image for HOG to reduce dimensions
        features = {}
        small = cv.resize(grey_frame, (64, 64))
        hog_feats = hog(small, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm='L2-Hys')
        
        # Statistical summary of HOG
        features['hog_mean'] = np.mean(hog_feats)
        features['hog_std'] = np.std(hog_feats)
        features['hog_max'] = np.max(hog_feats)
        return features

    def _get_optical_flow_features(self, prev_grey, curr_grey):
        features = {}
        
        # Farneback Dense Optical Flow
        flow = cv.calcOpticalFlowFarneback(prev_grey, curr_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        
        features['flow_mag_mean'] = np.mean(mag)
        features['flow_mag_std'] = np.std(mag)
        features['flow_ang_mean'] = np.mean(ang)
        
        # Histogram of flow magnitude
        hist_mag, _ = np.histogram(mag, bins=5, range=(0, 10))
        hist_mag = hist_mag.astype(float) / (hist_mag.sum() + 1e-7)
        for i, val in enumerate(hist_mag):
            features[f'flow_mag_hist_{i}'] = val
            
        return features

    def _process_video(self, row: dict):
        video_id = row['index']
        video_path = row.get('clip_path', '')
        if not video_path: 
            return []

        cap = cv.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()
        
        if not frames:
            return []

        total_frames = len(frames)
        selected_indices = []
        strategy = self.config['sampling']['strategy']
        
        if strategy == 'uniform':
            n_frames = self.config['sampling']['n_frames']
            if total_frames <= n_frames:
                selected_indices = list(range(total_frames))
            else:
                selected_indices = np.linspace(0, total_frames - 1, n_frames).astype(int)
        elif strategy == 'random':
            n_frames = self.config['sampling']['n_frames']
            if total_frames <= n_frames:
                selected_indices = list(range(total_frames))
            else:
                indices = np.random.choice(total_frames, n_frames, replace=False)
                selected_indices = np.sort(indices)
        else: # dense
            skip = self.config['sampling']['frame_skip']
            selected_indices = range(0, total_frames, skip)

        video_features = []
        prev_grey_frame = None
        # Should create new instance per video to avoid state bleed
        temporal_extractor = TemporalFeatureExtractor(5)

        for idx in selected_indices:
            frame = frames[idx]
            frame = self._preprocess_frame(frame)
            grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
            frame_feat = {'index': video_id}
            
            # Spatial Features
            frame_feat.update(self._get_frame_color_features(frame))
            frame_feat.update(self._get_frame_glcm_features(grey_frame))
            frame_feat.update(self._lbp_features(grey_frame))
            frame_feat.update(self._get_gabor_features(grey_frame))
            frame_feat.update(self._get_canny_features(grey_frame))
            frame_feat.update(self._get_contour_features(grey_frame))
            frame_feat.update(self._get_hog_features(grey_frame))
            
            # Temporal/Motion Features
            if prev_grey_frame is not None:
                motion_feat = self._get_optical_flow_features(prev_grey_frame, grey_frame)
                temporal_stats = temporal_extractor.update(motion_feat)
                frame_feat.update(motion_feat) 
                frame_feat.update(temporal_stats)
            else:
                # First frame, zero motion
                motion_feat = self._get_optical_flow_features(grey_frame, grey_frame) # Zero flow basically
                # Since _get_optical_flow calculates flow between two frames, for first frame we can pass SAME frame -> 0 flow
                temporal_stats = temporal_extractor.update(motion_feat)
                frame_feat.update(motion_feat)
                frame_feat.update(temporal_stats)
                
            video_features.append(frame_feat)
            prev_grey_frame = grey_frame

        return video_features
    
    def process_dataset(self, df: pd.DataFrame, is_test=False):
        print(f"Processing {len(df)} videos with {self.config['n_jobs']} jobs...")
        rows = df.reset_index().to_dict('records')
        
        # Using joblib backend 'threading' might be safer for OpenCV which releases GIL? 
        # But 'loky' (default) is safer for process isolation.
        nested_results = Parallel(n_jobs=self.config['n_jobs'])(delayed(self._process_video)(row) for row in rows)
        
        flatten_results = [item for sub_list in nested_results if sub_list for item in sub_list]
        feature_df = pd.DataFrame(flatten_results)
        
        feature_names = [col for col in feature_df.columns if col not in ['index', 'encoded_label']]
        # Fill NaNs
        feature_df[feature_names] = feature_df[feature_names].fillna(0)
        feature_df[feature_names] = feature_df[feature_names].replace([np.inf, -np.inf], 0)
        
        return feature_df

    def aggregate_features(self, feature_df: pd.DataFrame):
        """Aggregate frame-level features to video-level features."""
        if feature_df.empty:
            return pd.DataFrame()
            
        # Group by video index
        # We compute mean, std, min, max for each numerical feature
        agg_funcs = ['mean', 'std', 'min', 'max'] 
        
        # Drop non-numeric for aggregation
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        if 'index' in numeric_cols: numeric_cols.remove('index')
        if 'encoded_label' in numeric_cols: numeric_cols.remove('encoded_label')
        
        agg_dict = {col: agg_funcs for col in numeric_cols}
        
        # We need 'index' to group
        grouped = feature_df.groupby('index')
        
        agg_df = grouped[numeric_cols].agg(agg_funcs)
        
        # Flatten MultiIndex columns
        agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
        
        return agg_df

    def fit_transform_scaler(self, df, is_test=False):
        # Helper to scale features
        feature_names = [col for col in df.columns if col not in ['index', 'encoded_label']]
        
        # Check if scalar is initialized (it is in init)
        
        if is_test:
            # Handle unseen columns in test? 
            # Ensure columns match scaler? 
            # For now assume consistent feature extraction
            df[feature_names] = self.scaler.transform(df[feature_names])
        else:
            df[feature_names] = self.scaler.fit_transform(df[feature_names])
        return df
