import requests

def get_coordinates(city):
    url = f"https://nominatim.openstreetmap.org/search?city={city}&format=json"
    headers = {
    "User-Agent": "CropDiseaseApp/1.0 (princygarg2004@gmail.com)"
}
    response = requests.get(url, headers=headers)

    try:
        data = response.json()
    except ValueError:
        print("Error: Could not decode JSON from the response.")
        print(response.text)
        return None, None

    if data:
        return float(data[0]["lat"]), float(data[0]["lon"])
    return None, None