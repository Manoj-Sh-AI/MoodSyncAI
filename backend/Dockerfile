# Backend (Node.js)
FROM node:18-alpine
# Add tini for proper process management
RUN apk add --no-cache tini
WORKDIR /app
COPY package.json ./
RUN npm install --production
COPY src ./src
ENV PORT=8080
EXPOSE 8080
# Run node under tini
CMD ["/sbin/tini", "--", "node", "src/index.js"]
