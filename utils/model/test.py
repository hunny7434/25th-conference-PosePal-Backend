from utils.model.Rocket import RocketTransformerClassifier
import pickle

# Load from your old pickle (the one referencing __main__)
with open("utils/model/lateralraise_fin.pkl", "rb") as f:
    old_model = pickle.load(f)

# Now save again, but with the correct module path
with open("utils/model/lateralraise_fin.pkl", "wb") as f:
    pickle.dump(old_model, f)