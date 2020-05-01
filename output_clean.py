import os

for filename in os.listdir("output"):
    if filename.endswith(".json"):
        # Open json files and read into dict
        os.remove(os.path.join("output", filename))
        print(f"Removed {filename} before experiments")
