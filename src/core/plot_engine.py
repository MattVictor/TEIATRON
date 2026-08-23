import math
import numpy as np

class PlotEngine:
    @staticmethod
    def get_plot_data(model, **kwargs):
        class_name = model.__class__.__name__
        
        if class_name == "MinDistanceClassifier":
            return PlotEngine._plot_min_distance(model, **kwargs)
        elif class_name == "MaxDistanceClassifier":
            return {"empty_legends": ["Critério: Minimização da Distância Máxima"]}
        elif class_name == "PerceptronClassifier":
            return PlotEngine._plot_perceptron(model, **kwargs)
        elif class_name == "OptimalBayesMAP" or class_name == "NaiveBayesMAP":
            return PlotEngine._plot_bayes(model, **kwargs)
        elif class_name == "SVMClassifier":
            return PlotEngine._plot_svm(model, **kwargs)
        else:
            return {}

    @staticmethod
    def _plot_min_distance(model, **kwargs):
        x_key = kwargs.get('x_key')
        y_key = kwargs.get('y_key')
        if not x_key or not y_key: return {}
        
        selected_features = getattr(model, 'selected_features', ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"])
        if x_key not in selected_features or y_key not in selected_features:
            return {"empty_legends": ["O modelo não utilizou ambos os eixos selecionados."]}
            
        idx_x = selected_features.index(x_key)
        idx_y = selected_features.index(y_key)
        
        points = []
        for c_name, coords in model.centroids.items():
            if len(coords) > max(idx_x, idx_y):
                cx, cy = coords[idx_x], coords[idx_y]
                points.append({"x": cx, "y": cy, "name": f"Centróide {c_name}", "symbol": "x", "size": 15, "color": "w"})
            
        lines = []
        if len(model.classes_trained) == 2:
            c1_coords = model.centroids[model.classes_trained[0]]
            c2_coords = model.centroids[model.classes_trained[1]]
            if len(c1_coords) > max(idx_x, idx_y) and len(c2_coords) > max(idx_x, idx_y):
                nx = c2_coords[idx_x] - c1_coords[idx_x]
                ny = c2_coords[idx_y] - c1_coords[idx_y]
                mx = (c1_coords[idx_x] + c2_coords[idx_x]) / 2.0
                my = (c1_coords[idx_y] + c2_coords[idx_y]) / 2.0
                
                if ny != 0:
                    angulo_deg = math.degrees(math.atan2(-nx, ny))
                    a = -nx / ny
                    b = my - (a * mx)
                    sinal = '+' if b >= 0 else '-'
                    equacao = f"g(x) = {a:.2f}x {sinal} {abs(b):.2f}"
                else:
                    angulo_deg = 90
                    equacao = f"g(x) -> x = {mx:.2f}"
                lines.append({"angle": angulo_deg, "pos": (mx, my), "name": equacao})
            
        return {"points": points, "lines": lines}

    @staticmethod
    def _plot_perceptron(model, **kwargs):
        x_key = kwargs.get('x_key')
        y_key = kwargs.get('y_key')
        dataset = kwargs.get('dataset')
        if not x_key or not y_key or not dataset: return {}
        
        selected_features = getattr(model, 'selected_features', ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"])
        b = model.pesos[0]
        
        if x_key not in selected_features or y_key not in selected_features:
            return {"empty_legends": ["O modelo não utilizou ambos os eixos selecionados."]}
            
        w_x = model.pesos[selected_features.index(x_key) + 1]
        w_y = model.pesos[selected_features.index(y_key) + 1]
        
        effective_bias = b
        for i, k in enumerate(selected_features):
            if k != x_key and k != y_key:
                dados_coluna = dataset.get(k, [])
                if len(dados_coluna) > 0:
                    media_coluna = sum(dados_coluna) / len(dados_coluna)
                    effective_bias += model.pesos[i + 1] * media_coluna
                    
        lines = []
        if w_y != 0:
            a = -w_x / w_y
            intercept = -effective_bias / w_y
            angulo_deg = math.degrees(math.atan(a))
            mx, my = 0, intercept 
            sinal_wy = "+" if w_y >= 0 else "-"
            sinal_b = "+" if effective_bias >= 0 else "-"
            equacao = f"{w_x:.2f}x {sinal_wy} {abs(w_y):.2f}y {sinal_b} {abs(effective_bias):.2f} = 0"
        else:
            angulo_deg = 90
            mx, my = -effective_bias / w_x if w_x != 0 else 0, 0
            sinal_b = "+" if effective_bias >= 0 else "-"
            equacao = f"{w_x:.2f}x {sinal_b} {abs(effective_bias):.2f} = 0"
            
        lines.append({"angle": angulo_deg, "pos": (mx, my), "name": f"Fronteira 2D: {equacao}"})
        return {"lines": lines}

    @staticmethod
    def _plot_bayes(model, **kwargs):
        x_key = kwargs.get('x_key')
        y_key = kwargs.get('y_key')
        dataset = kwargs.get('dataset')
        x_data = kwargs.get('x_data')
        y_data = kwargs.get('y_data')
        
        selected_features = getattr(model, 'selected_features', ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"])
        if not x_key or not y_key or not dataset or len(model.classes) < 2: 
            return {}
            
        if x_key not in selected_features or y_key not in selected_features:
            return {"empty_legends": ["O modelo não utilizou ambos os eixos selecionados."]}
            
        idx_x = selected_features.index(x_key)
        idx_y = selected_features.index(y_key)
        hid_idx = [i for i in range(len(selected_features)) if i not in [idx_x, idx_y]]
        
        hid_vals = [np.mean(dataset.get(selected_features[i], [])) for i in hid_idx]
        
        res = 200
        rx = max(x_data) - min(x_data)
        ry = max(y_data) - min(y_data)
        x_min, x_max = min(x_data) - rx, max(x_data) + rx
        y_min, y_max = min(y_data) - ry, max(y_data) + ry
        
        xi = np.linspace(x_min, x_max, res)
        yi = np.linspace(y_min, y_max, res)
        
        contours = []
        is_naive = model.__class__.__name__ == "NaiveBayesMAP"
        
        for i in range(len(model.classes)):
            for j in range(i + 1, len(model.classes)):
                c1, c2 = model.classes[i], model.classes[j]
                
                if is_naive:
                    m_i = model.parameters[c1]['mean']
                    var_i = model.parameters[c1]['var']
                    inv_cov_i = np.diag(1.0 / var_i)
                    m_j = model.parameters[c2]['mean']
                    var_j = model.parameters[c2]['var']
                    inv_cov_j = np.diag(1.0 / var_j)
                    prior_i = model.priors[c1]
                    prior_j = model.priors[c2]
                    
                    W = -0.5 * (inv_cov_i - inv_cov_j)
                    w = np.dot(inv_cov_i, m_i) - np.dot(inv_cov_j, m_j)
                    term_w0_1 = -0.5 * (np.dot(np.dot(m_i.T, inv_cov_i), m_i) - np.dot(np.dot(m_j.T, inv_cov_j), m_j))
                    term_w0_2 = -0.5 * np.sum(np.log(var_i)) + 0.5 * np.sum(np.log(var_j))
                    term_w0_3 = np.log(prior_i) - np.log(prior_j)
                    w0 = term_w0_1 + term_w0_2 + term_w0_3
                else:
                    m_i = model.parameters[c1]['mean']
                    cov_i = model.parameters[c1]['cov']
                    try: inv_cov_i = np.linalg.inv(cov_i)
                    except: inv_cov_i = np.eye(len(m_i))
                    m_j = model.parameters[c2]['mean']
                    cov_j = model.parameters[c2]['cov']
                    try: inv_cov_j = np.linalg.inv(cov_j)
                    except: inv_cov_j = np.eye(len(m_j))
                    prior_i = model.priors[c1]
                    prior_j = model.priors[c2]
                    
                    W = -0.5 * (inv_cov_i - inv_cov_j)
                    w = np.dot(inv_cov_i, m_i) - np.dot(inv_cov_j, m_j)
                    term_w0_1 = -0.5 * (np.dot(np.dot(m_i.T, inv_cov_i), m_i) - np.dot(np.dot(m_j.T, inv_cov_j), m_j))
                    det_cov_i = max(np.linalg.det(cov_i), 1e-10)
                    det_cov_j = max(np.linalg.det(cov_j), 1e-10)
                    term_w0_2 = -0.5 * np.log(det_cov_i / det_cov_j)
                    term_w0_3 = np.log(prior_i) - np.log(prior_j)
                    w0 = term_w0_1 + term_w0_2 + term_w0_3
                
                Z = np.zeros((res, res))
                for idx_i, xv in enumerate(xi):
                    for idx_j, yv in enumerate(yi):
                        vec = np.zeros(len(selected_features))
                        vec[idx_x] = xv
                        vec[idx_y] = yv
                        for k, h_i in enumerate(hid_idx):
                            vec[h_i] = hid_vals[k]
                        val = np.dot(np.dot(vec.T, W), vec) + np.dot(w.T, vec) + w0
                        Z[idx_i, idx_j] = val
                        
                contours.append({
                    "Z": Z, "level": 0.0, "x_min": x_min, "x_max": x_max, 
                    "y_min": y_min, "y_max": y_max, "res": res, 
                    "name": f"Fronteira {c1} x {c2}"
                })
                
        return {"contours": contours}

    @staticmethod
    def _plot_svm(model, **kwargs):
        if model.model is None: return {}
        
        x_key = kwargs.get('x_key')
        y_key = kwargs.get('y_key')
        dataset = kwargs.get('dataset')
        x_data = kwargs.get('x_data')
        y_data = kwargs.get('y_data')
        
        selected_features = getattr(model, 'selected_features', ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"])
        if not x_key or not y_key or not dataset: return {}
        
        if x_key not in selected_features or y_key not in selected_features:
            return {"empty_legends": ["O modelo não utilizou ambos os eixos selecionados."]}
            
        idx_x = selected_features.index(x_key)
        idx_y = selected_features.index(y_key)
        hid_idx = [i for i in range(len(selected_features)) if i not in [idx_x, idx_y]]
        
        hid_vals = [np.mean(dataset.get(selected_features[i], [])) for i in hid_idx]
        
        res = 200
        rx = max(x_data) - min(x_data)
        ry = max(y_data) - min(y_data)
        x_min, x_max = min(x_data) - rx, max(x_data) + rx
        y_min, y_max = min(y_data) - ry, max(y_data) + ry
        
        xi = np.linspace(x_min, x_max, res)
        yi = np.linspace(y_min, y_max, res)
        
        grid_points = []
        for xv in xi:
            for yv in yi:
                vec = np.zeros(len(selected_features))
                vec[idx_x] = xv
                vec[idx_y] = yv
                for k, h_i in enumerate(hid_idx):
                    vec[h_i] = hid_vals[k]
                grid_points.append(vec)
                
        Z_flat = model.model.decision_function(grid_points)
        Z = Z_flat.reshape((res, res))
        
        points = []
        if hasattr(model.model, 'support_vectors_'):
            for idx, sv in enumerate(model.model.support_vectors_):
                points.append({
                    "x": sv[idx_x],
                    "y": sv[idx_y],
                    "name": "Vetor de Suporte" if idx == 0 else None,
                    "symbol": "o",
                    "size": 16,
                    "color": "#FFFF00" 
                })
        
        return {
            "contours": [{"Z": Z, "level": 0.0, "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max, "res": res, "name": "Fronteira SVM"}],
            "points": points
        }
