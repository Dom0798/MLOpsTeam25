from pydantic import BaseModel
from typing import Optional


class HomeFeatures(BaseModel):
    bathrooms: float
    bedrooms: float
    square_feet: float
    latitude: float
    longitude: float
    has_photo_Thumbnail: bool
    has_photo_Yes: bool
    pets_allowed_CatsDogs: bool
    pets_allowed_CatsDogsNone: bool
    pets_allowed_Dogs: bool
    pets_allowed_unknown: bool
