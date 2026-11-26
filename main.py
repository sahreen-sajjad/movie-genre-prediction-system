import argparse
import json
from pathlib import Path
import joblib
from src.data import load_data, split_data
from src.preprocessing import preprocess
from src.model import build_model
from src.evaluate import evaluate, save_confusion_matrix

def main():
    parser = argparse.ArgumentParser(description='Train or use a movie dialogue genre classifier')
    parser.add_argument('--split', choices=['grouped','original'], default='grouped')
    parser.add_argument('--predict', help='Predict genre for a dialogue using a saved model')
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    if args.predict is not None:
        model = joblib.load(args.output / 'model.joblib')
        print(model.predict([[preprocess(args.predict)]])[0])
        return
    train,test,val = split_data(load_data(), grouped=args.split == 'grouped')
    model = build_model()
    def features(frame):
        return frame['text'].map(preprocess).to_frame('tokens').to_numpy()
    model.fit(features(train), train['meta.genre'])
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for name, frame in [('validation',val),('test',test)]:
        predictions = model.predict(features(frame))
        metrics[name] = evaluate(frame['meta.genre'], predictions)
        save_confusion_matrix(frame['meta.genre'], predictions, args.output / f'{name}_confusion.png')
    (args.output / 'metrics.json').write_text(json.dumps(metrics,indent=2))
    joblib.dump(model,args.output / 'model.joblib')
    print(json.dumps(metrics,indent=2))

if __name__ == '__main__':
    main()
