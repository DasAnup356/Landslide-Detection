import numpy as np 
import pandas as pd 
from PIL import Image

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib 

import os 
import argparse 


def load_labels_csv(csv_path: str) -> pd.DataFrame: 

    df = pd.read_csv(csv_path) #expected Columns: [filename, label(0/1)]
    required = ['filename', 'label']
    for c in required: 
        if c not in df.columns:
            raise ValueError(f'CSV missing requried column: {c}')
    return df

def open_patch(patch_folder: str, 
               filename: str) -> Image.Image: 
    path = os.path.join(patch_folder, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Patch not found: {path}")
    img = Image.open(path).convert('RGB') 
    return img


def train_rf_from_csv(csv_path: str,
                      patch_folder: str,
                      model_out_path: str, 
                      test_size=0.2, 
                      random_state=42):
    #
    df = load_labels_csv(csv_path)

    #
    X_list = []
    y_list = []
    feature_names = None

    from utils.features import extract_features_from_patch
    
    for idx, row in df.iterrows():
        fn = row['filename']
        lbl = int(row['label'])
        try: 
            patch = open_patch(patch_folder, fn)
        except Exception as e:
            print(f"Warning: skipping {fn}: {e}")
            continue
   
        feats, names = extract_features_from_patch(patch)
        if feature_names is None:
            feature_names = names
        X_list.append(feats)
        y_list.append(lbl)

        if (idx+1)%1000 == 0: 
            print( f"processed {idx+1} patches" )
    
    X = np.vstack(X_list)
    y = np.array(y_list)
    print(  f"Feature matrix shape: {X.shape}" )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state, stratify=y)
    
    #
    clf = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1,
                                class_weight='balanced', random_state=random_state)
    clf.fit(X_train, y_train)
    
    #
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    #
    payload = {
        'model': clf,
        'feature_names': feature_names,
        'patch_w': X.shape [1],
    }
    joblib.dump(payload, model_out_path)
    print(f"Saved model to {model_out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train'], required=True)
    parser.add_argument('--csv', help ='labels csv for training')
    parser.add_argument('--patches_folder', help='folder with patch images')
    parser.add_argument('--model_out', help='where to save model (joblib)')
    args = parser.parse_args() 

    if args.mode == 'train': 
        if not args.csv or not args.patches_folder or not args.model_out:
            parser.error('Training requires --csv, --patch_folder and --model_out')

        train_rf_from_csv(args.csv, args.patches_folder, args.model_out)

if __name__ == "__main__":
    main()

