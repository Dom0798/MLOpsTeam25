# Build the Docker image
docker build -t apartment-rent-api .

# Run the Docker container
docker run -p 5000:5000 apartment-rent-api

# Test the API
## Using curl
curl -Method Post -Uri "http://localhost:5000/predict-price" -Headers @{ "Content-Type" = "application/json" } -Body '{"bathrooms": 1, "bedrooms": 1, "square_feet": 120, "latitude": 25, "longitude": -70, "has_photo_Thumbnail": 1, "has_photo_Yes": 0, "pets_allowed_CatsDogs": 1, "pets_allowed_CatsDogsNone": 1, "pets_allowed_Dogs": 0, "pets_allowed_unknown": 0}'
## Using Postman
1. Select POST as mode
2. Place http://localhost:5000/predict-price in URL
3. In Body, select raw and use the input: {"bathrooms": 1, "bedrooms": 1, "square_feet": 120, "latitude": 25, "longitude": -70, "has_photo_Thumbnail": 1, "has_photo_Yes": 0, "pets_allowed_CatsDogs": 1, "pets_allowed_CatsDogsNone": 1, "pets_allowed_Dogs": 0, "pets_allowed_unknown": 0}