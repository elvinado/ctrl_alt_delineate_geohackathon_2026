"""Compose the versioned API from resource and shared routers."""

from fastapi import APIRouter

from app.resources_api import router as resource_router
from app.format_info import router as format_router
from app.settings_api import router as settings_router
from app.processing_jobs import router as processing_jobs_router
from app.model_releases import router as model_releases_router
from app.scripts_api import router as scripts_router
from app.geochat.api import router as geochat_router
from app.geobody_api import router as geobody_router

api_router = APIRouter()

# Pipeline stages are architectural concepts, not URL namespaces. Routes are
# organized around resources such as projects, surveys, wells, and datasets.
api_router.include_router(resource_router, tags=["resources"])
api_router.include_router(settings_router, prefix="/settings", tags=["settings"])
api_router.include_router(format_router, tags=["capabilities"])
api_router.include_router(scripts_router, tags=["processing scripts"])
api_router.include_router(processing_jobs_router, tags=["processing jobs"])
api_router.include_router(model_releases_router, tags=["model releases"])

api_router.include_router(geochat_router, prefix="/geochat", tags=["geochat"])
api_router.include_router(geobody_router, prefix="/geobodies", tags=["geobodies"])
