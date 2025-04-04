from pathlib import Path
from xlit import Xlit

if __name__ == "__main__":
    # Paths
    data_path = Path("data") / "transcribed.txt"
    exp_dir = Path("experiments")
    config_path = Path("data") / "config.json"
    model_path = exp_dir / "best_model.pth"

    # Training
    xlit = Xlit.from_scratch()
    xlit.train(data_path=data_path, exp_dir=exp_dir)

    # Prediction
    xlit_loaded = Xlit.from_pretrained(model_path, config_path)
    test_text = "হ্যালো"
    result = xlit_loaded.predict(test_text)
    print(result)
