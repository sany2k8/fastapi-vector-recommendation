from fastapi import APIRouter

from app.api.v1 import admin, interactions, products, recommendations, search

api_router = APIRouter()
api_router.include_router(products.router)
api_router.include_router(search.router)
api_router.include_router(recommendations.router)
api_router.include_router(interactions.router)
api_router.include_router(admin.router)

__all__ = ["api_router"]
