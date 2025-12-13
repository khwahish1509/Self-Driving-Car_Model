import matplotlib.pyplot as plt

# Reviewing and Balancing the Dataset
angles = []
image_paths = []

with open("driving_log.csv") as f:
    for line in f:
        parts = line.strip().split(",")
        image_paths.append(parts[0].strip())
        angle = float(parts[3])
        angles.append(angle)

plt.hist(angles, bins=50)
plt.xlabel("Steering Angle Values")
plt.ylabel("Frequency of Images")
plt.title("Steering Angle Distribution Before Balancing")
plt.show()
