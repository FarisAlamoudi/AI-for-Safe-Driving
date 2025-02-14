import cv2
import numpy as np
import imutils
import os
import glob

# Slope of line function
def Slope(a, b, c, d):
    if c - a == 0:
        return float('inf')  # Avoid division by zero for vertical lines
    return (d - b) / (c - a)

# Specify the folder containing your images
image_folder = "/Users/farisal-amoudi/Desktop/Fall 24/SD1/Testing for seatbelt/YoloImages"  # Replace with the path to your folder

# Get a list of all image files in the folder (e.g., jpg, png, jpeg)
image_paths = glob.glob(os.path.join(image_folder, "*.jpg")) + \
              glob.glob(os.path.join(image_folder, "*.png")) + \
              glob.glob(os.path.join(image_folder, "*.jpeg"))

# Check if image paths were found
if not image_paths:
    print(f"No images found in {image_folder}")
else:
    print(f"Found {len(image_paths)} images in {image_folder}")

# Loop through each image in the folder
for image_path in image_paths:
    # Reading Image
    beltframe = cv2.imread(image_path)

    # Check if the image was successfully loaded
    if beltframe is None:
        print(f"Error loading image {image_path}")
        continue

    # Resizing The Image
    beltframe = imutils.resize(beltframe, height=800)

    # Converting To GrayScale
    beltgray = cv2.cvtColor(beltframe, cv2.COLOR_BGR2GRAY)

    # No Belt Detected Yet
    belt = False

    # Blurring The Image For Smoothness
    blur = cv2.blur(beltgray, (1, 1))

    # Converting Image To Edges
    edges = cv2.Canny(blur, 50, 400)

    # Previous Line Slope
    ps = 0

    # Previous Line Co-ordinates
    px1, py1, px2, py2 = 0, 0, 0, 0

    # Extracting Lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 270, 30, maxLineGap=20, minLineLength=170)

    # If "lines" Is Not Empty
    if lines is not None:
        # Loop line by line
        for line in lines:
            # Co-ordinates Of Current Line
            x1, y1, x2, y2 = line[0]

            # Slope Of Current Line
            s = Slope(x1, y1, x2, y2)

            # If Current Line's Slope Is Greater Than 0.7 And Less Than 2
            if (0.7 < abs(s) < 2):
                # And Previous Line's Slope Is Within 0.7 To 2
                if (0.7 < abs(ps) < 2):
                    # And Both The Lines Are Not Too Far From Each Other
                    if (((abs(x1 - px1) > 5) and (abs(x2 - px2) > 5)) or ((abs(y1 - py1) > 5) and (abs(y2 - py2) > 5))):
                        # Plot The Lines On "beltframe"
                        cv2.line(beltframe, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.line(beltframe, (px1, py1), (px2, py2), (0, 0, 255), 3)

                        # Belt Is Detected
                        print(f"Belt Detected in {image_path}")
                        belt = True

            # Update the previous slope and coordinates
            ps = s
            px1, py1, px2, py2 = line[0]

    if not belt:
        print(f"No Seatbelt detected in {image_path}")

    # Handle folder creation and file conflicts
    result_folder = "result"
    
    # Check if "result" is a file, not a directory
    if os.path.exists(result_folder) and not os.path.isdir(result_folder):
        print(f"Error: A file named 'result' exists, but a directory is needed.")
        continue  # Skip processing if there's a conflict

    # Create result directory if it doesn't exist
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Define the output filename
    output_filename = os.path.join(result_folder, os.path.basename(image_path))

    # Save the processed image to the output folder
    if cv2.imwrite(output_filename, beltframe):
        print(f"Image saved to: {output_filename}")
    else:
        print(f"Failed to save image to {output_filename}")

    # Optional text file write for testing purposes
    test_file = os.path.join(result_folder, "test.txt")
    with open(test_file, "w") as f:
        f.write("Test write complete.\n")
        print(f"Test file written to: {test_file}")

    # If no lines were detected
    if lines is None:
        print(f"No lines detected in {image_path}")
    else:
        print(f"Detected {len(lines)} lines in {image_path}")

# Close all OpenCV windows
cv2.destroyAllWindows()
