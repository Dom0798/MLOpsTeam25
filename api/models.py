from pydantic import BaseModel
from typing import Optional


class HomeFeatures(BaseModel):
    bathrooms: float
    bedrooms: float
    square_feet: float
    latitude: float
    longitude: float
    has_thumbnail: bool
    has_photo: bool
    allows_pets: Optional[bool] = None
