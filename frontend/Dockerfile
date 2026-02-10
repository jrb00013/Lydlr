FROM node:18-alpine

WORKDIR /app

# Install wget for healthchecks
RUN apk add --no-cache wget curl

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy app files
COPY . .

# Expose port
EXPOSE 3000

# Start development server
CMD ["npm", "start"]

