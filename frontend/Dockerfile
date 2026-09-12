# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# TRACE frontend — production image
# Build stage compiles the Vite app; runtime stage is just static nginx.
#
# VITE_API_URL is baked in at BUILD time (Vite inlines env vars into the JS
# bundle), so it must be passed as a --build-arg, not a runtime -e flag.
# Default here ("/api") assumes nginx.conf proxies /api/* to the backend
# container — see nginx.conf and docker-compose.yml.
# ---------------------------------------------------------------------------

FROM node:20-alpine AS build
WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
ARG VITE_API_URL=/api
ENV VITE_API_URL=${VITE_API_URL}
RUN npm run build

FROM nginx:1.27-alpine AS runtime
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost:80 || exit 1

CMD ["nginx", "-g", "daemon off;"]