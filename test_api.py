import requests

# API endpoint URL
API_URL = "http://127.0.0.1:5000/process_image"  # Replace with your PC's local IP and port if different

# Path to the image you want to test
IMAGE_PATH = "okay.jpg"  # Replace with the path to your test image

# Optional parameters
confidence = 0.5
crowd_density = 100

# Send the image to the API
def test_api():
    try:
        # Open the image file in binary mode
        with open(IMAGE_PATH, "rb") as image_file:
            # Prepare the payload
            files = {"image": image_file}
            data = {
                "confidence": confidence,
                "crowd_density": crowd_density
            }

            # Send POST request to the API
            response = requests.post(API_URL, files=files, data=data)

            # Check the response status
            if response.status_code == 200:
                # Save the output image to a file
                output_file = "output_image.jpg"
                with open(output_file, "wb") as f:
                    f.write(response.content)
                print(f"Processed image saved as {output_file}")
            else:
                print(f"Error: {response.status_code}, {response.text}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Run the test
if __name__ == "__main__":
    test_api()
