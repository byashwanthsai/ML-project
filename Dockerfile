FROM node:22-alpine AS frontend-build
WORKDIR /workspace/frontend
COPY frontend/package.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt
COPY backend/app /app/backend/app
COPY backend/models/artifacts.joblib /app/backend/models/artifacts.joblib
COPY backend/models/metrics.json /app/backend/models/metrics.json
COPY --from=frontend-build /workspace/frontend/dist /app/frontend/dist
ENV PYTHONPATH=/app/backend PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
