# API Documentation

This document provides instructions for testing each endpoint using Postman.

## Base URL
```
http://localhost:3000
```

## Endpoints

### 1. Create Profile
**Endpoint**: `POST /profile`

**Request Body**:
```json
{
  "name": "John Doe",
  "age": 25,
  "eye_color": "blue",
  "hair_type": "straight",
  "image_url": "https://example.com/image.jpg",
  "additional_attributes": {
    "interests": ["hiking", "photography"],
    "location": "New York"
  }
}
```

**Success Response** (201):
```json
{
  "message": "Profile created successfully",
  "profile_id": "John Doe"
}
```

### 2. Search Profiles
**Endpoint**: `POST /search`

**Request Body**:
```json
{
  "text_query": "looking for someone who loves hiking",
  "filters": {
    "age": 25,
    "eye_color": "blue"
  },
  "page": 1,
  "per_page": 10
}
```

**Success Response** (200):
```json
{
  "results": [
    {
      "profile_id": "John Doe",
      "similarity_score": 0.85,
      "metadata": {
        "name": "John Doe",
        "age": 25,
        "eye_color": "blue",
        "hair_type": "straight",
        "interests": ["hiking", "photography"]
      }
    }
  ],
  "meta": {
    "total_candidates": 100,
    "filtered_count": 10,
    "max_score": 0.95,
    "mean_score": 0.75,
    "threshold": 0.5,
    "query": "looking for someone who loves hiking"
  }
}
```

### 3. Profile Similarities
**Endpoint**: `POST /profile-similarities`

**Request Body**:
```json
{
  "profile_id": "John Doe",
  "target_profile_id": "Jane Smith"
}
```

**Success Response** (200):
```json
{
  "profiles": {
    "profile1": {
      "name": "John Doe",
      "attributes": {
        "age": 25,
        "eye_color": "blue",
        "hair_type": "straight",
        "interests": ["hiking", "photography"]
      }
    },
    "profile2": {
      "name": "Jane Smith",
      "attributes": {
        "age": 28,
        "eye_color": "green",
        "hair_type": "curly",
        "interests": ["hiking", "painting"]
      }
    }
  },
  "similarity_score": 0.75,
  "personalized_message": "John and Jane both love hiking! Think of the wonderful adventures that lie ahead!"
}
```

### 4. Store Agent Response
**Endpoint**: `POST /agent-response`

**Request Body**:
```json
{
  "userId": "user123",
  "agentId": "agent456",
  "question": "What's your favorite hobby?",
  "response": "I love hiking and photography!",
  "userMetadata": {
    "name": "John Doe",
    "age": 25,
    "additionalInfo": {
      "location": "New York",
      "interests": ["hiking", "photography"]
    }
  }
}
```

**Success Response** (201):
```json
{
  "message": "Response stored successfully",
  "responseId": "agent456-user123-1640995200000"
}
```

### 5. Search Similar Responses
**Endpoint**: `POST /search-similar-responses`

**Request Body**:
```json
{
  "agentId": "agent456",
  "response": "I love hiking and outdoor photography",
  "filters": {
    "userAge": { "$gte": 20, "$lte": 30 }
  },
  "page": 1,
  "per_page": 10
}
```

**Success Response** (200):
```json
{
  "results": [
    {
      "responseId": "agent456-user123-1640995200000",
      "similarity_score": 0.88,
      "metadata": {
        "userId": "user123",
        "userName": "John Doe",
        "userAge": 25,
        "response": "I love hiking and photography!",
        "timestamp": "2023-12-31T12:00:00Z"
      },
      "relative_score": 0.92
    }
  ],
  "meta": {
    "agentId": "agent456",
    "total_matches": 50,
    "filtered_count": 10,
    "threshold": 0.5,
    "query_response": "I love hiking and outdoor photography"
  }
}
```

### 6. Health Check
**Endpoint**: `GET /health`

**Success Response** (200):
```json
{
  "status": "healthy",
  "services": {
    "api": "up",
    "redis": "up",
    "pinecone": "up",
    "clip_api": "up"
  }
}
```

## Error Handling

All endpoints return appropriate error responses in the following format:

```json
{
  "error": "Error message description",
  "meta": {
    "timestamp": "2023-12-31T12:00:00Z"
  }
}
```

## Testing Tips

1. Ensure all required services (Redis, Pinecone, CLIP API) are running before testing
2. Use valid image URLs for profile creation
3. Keep text queries concise and relevant
4. Monitor response times and cache usage
5. Test with various filter combinations
6. Verify error handling by providing invalid inputs

## Environment Variables

Make sure to set up the following environment variables in js directory:

```
REDIS_HOST=localhost
REDIS_PORT=6379
CLIP_API_URL=http://localhost:5002
PINECONE_API_KEY=your_pinecone_api_key
HYPERBOLIC_API_KEY=your_hyperbolic_api_key
```
