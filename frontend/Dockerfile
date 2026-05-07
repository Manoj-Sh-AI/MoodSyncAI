# Build stage
FROM node:18-alpine AS build
# Allow build-time arg for the API URL
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL
WORKDIR /app
COPY package.json ./
# If you later add a lockfile, copy it here and use `npm ci`
RUN npm install
COPY . .
# Use the env var during the build
RUN npm run build

# Serve stage
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
