const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const axios = require('axios');
const Redis = require('ioredis');
const { Pinecone } = require('@pinecone-database/pinecone');
const sharp = require('sharp');
const Joi = require('joi');

dotenv.config();

const app = express();

app.use(cors());
app.use(express.json({ limit: '50mb' })); 

const redis = new Redis({
  host: process.env.REDIS_HOST || 'localhost',
  port: process.env.REDIS_PORT || 6379
});

const CLIP_API_URL = process.env.CLIP_API_URL || 'http://localhost:5002';
const HYPERBOLIC_API_URL = 'https://api.hyperbolic.xyz/v1/chat/completions';
const HYPERBOLIC_API_KEY = process.env.HYPERBOLIC_API_KEY;

const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY, 
});

const profileIndex = pc.index('swifey');
const agentIndex = pc.index('ai-agent');

const profileSchema = Joi.object({
  name: Joi.string().required(),
  age: Joi.number().integer().min(0).required(),
  eye_color: Joi.string().required(),
  hair_type: Joi.string().required(),
  image_url: Joi.string().uri().required(),
  additional_attributes: Joi.object().default({})
});

const searchSchema = Joi.object({
  text_query: Joi.string().required(),
  filters: Joi.object().default({}),
  page: Joi.number().integer().min(1).default(1),
  per_page: Joi.number().integer().min(1).max(100).default(10)
});

const profileSimilaritySchema = Joi.object({
  profile_id: Joi.string().required(),
  target_profile_id: Joi.string().required()
});

const agentResponseSchema = Joi.object({
  userId: Joi.string().required(),
  agentId: Joi.string().required(),
  question: Joi.string().required(),
  response: Joi.string().required(),
  userMetadata: Joi.object({
    name: Joi.string().required(),
    age: Joi.number().integer().min(0).required(),
    additionalInfo: Joi.object().default({})
  }).required()
});

const agentSearchSchema = Joi.object({
  agentId: Joi.string().required(),
  response: Joi.string().required(),
  filters: Joi.object().default({}),
  page: Joi.number().integer().min(1).default(1),
  per_page: Joi.number().integer().min(1).max(100).default(10)
});

async function downloadImage(url) {
  try {
    const response = await axios({
      url,
      method: 'GET',
      responseType: 'arraybuffer',
      headers: {
        'Accept': 'image/*',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
      },
      maxRedirects: 5
    });

    const contentType = response.headers['content-type'];
    if (!contentType.startsWith('image/')) {
      throw new Error(`Invalid content type: ${contentType}`);
    }

    const imageBuffer = await sharp(response.data)
      .resize(224, 224, {
        fit: 'cover',
        position: 'center'
      })
      .toBuffer();

    return imageBuffer;
  } catch (error) {
    console.error('Download error:', error);
    throw new Error(`Error downloading image: ${error.message}`);
  }
}

async function getClipEmbeddings(type, input) {
  try {
    let payload;
    
    if (type === 'image') {
      const base64Image = input.toString('base64');
      payload = {
        type: 'image_base64',
        input: `data:image/jpeg;base64,${base64Image}`
      };
    } else {
      payload = {
        type: 'text',
        input: input
      };
    }

    const response = await axios.post(
      `${CLIP_API_URL}/embeddings`,
      payload,
      {
        headers: {
          'Content-Type': 'application/json'
        },
        maxBodyLength: Infinity,
        maxContentLength: Infinity
      }
    );

    if (!response.data || !response.data.embeddings) {
      throw new Error('Invalid response from CLIP API');
    }

    return response.data.embeddings;
  } catch (error) {
    console.error('CLIP API Error:', error.response?.data || error.message);
    throw new Error(`Error getting CLIP embeddings: ${error.message}`);
  }
}

function cosineSimilarity(vector1, vector2) {
  if (vector1.length !== vector2.length) {
    throw new Error('Vectors must have the same length');
  }

  let dotProduct = 0;
  let norm1 = 0;
  let norm2 = 0;

  for (let i = 0; i < vector1.length; i++) {
    dotProduct += vector1[i] * vector2[i];
    norm1 += vector1[i] * vector1[i];
    norm2 += vector2[i] * vector2[i];
  }

  return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
}

function findCommonInterests(profile1Meta, profile2Meta) {
  const interests1 = new Set(profile1Meta.additional_attributes?.interests || []);
  const interests2 = new Set(profile2Meta.additional_attributes?.interests || []);
  return Array.from(interests1).filter(interest => interests2.has(interest));
}

async function generateSimilarityMessage(profile1Meta, profile2Meta, similarityScore) {
  try {
    const commonInterests = findCommonInterests(profile1Meta, profile2Meta);
    let sharedInterestPart = '';
    if (commonInterests.length > 0) {
      sharedInterestPart = commonInterests.length === 1 ? 
        `${profile1Meta.name} and ${profile2Meta.name} both love ${commonInterests[0]}. ` : 
        `${profile1Meta.name} and ${profile2Meta.name} share common interests in ${commonInterests.slice(0, -1).join(', ')} and ${commonInterests.slice(-1)}. `;
    }

    const prompt = `
    Let's craft a message for ${profile2Meta.name} and me that's not just cheesy, but also heartfelt and detailed. Here's what you need to know about them:
    - ${profile1Meta.name}: ${JSON.stringify(profile1Meta)}
    - ${profile2Meta.name}: ${JSON.stringify(profile2Meta)}
    They've just found out they ${commonInterests.length > 0 ? `share a passion for ${commonInterests.length > 1 ? 'several things' : commonInterests[0]}` : 'have something incredible in common'}. We want a message that's rich in puns, playful jokes, and celebrates their shared interest in a way that's both funny and touching. Include: "${sharedInterestPart}Think of the wonderful adventures that lie ahead!"
    Aim for a message that's under 300 characters, packed with humor and warmth, making it impossible for them not to smile.`;

    const response = await axios.post(
      HYPERBOLIC_API_URL,
      {
        messages: [
          {
            role: "user",
            content: prompt
          }
        ],
        model: "meta-llama/Llama-3.3-70B-Instruct",
        max_tokens: 1024,
        temperature: 0.7,
        top_p: 0.9
      },
      {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${HYPERBOLIC_API_KEY}`
        }
      }
    );

    let message = response.data.choices[0].message.content;
    
    return message;
  } catch (error) {
    console.error('Error generating message:', error);
    return `${profile2Meta.name} and you seem to have a lot in common. You're off to a great start!`;
  }
}

app.post('/profile', async (req, res) => {
  try {
    const { error, value } = profileSchema.validate(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    const imageBuffer = await downloadImage(value.image_url);
    const imageEmbedding = await getClipEmbeddings('image', imageBuffer);
    
    const metadata = {
      name: value.name,
      age: value.age,
      eye_color: value.eye_color,
      hair_type: value.hair_type,
      image_url: value.image_url,
      ...value.additional_attributes,
      created_at: new Date().toISOString()
    };

    await profileIndex.upsert([{
      id: value.name,
      values: imageEmbedding,
      metadata: metadata
    }]);

    res.status(201).json({
      message: 'Profile created successfully',
      profile_id: value.name
    });

  } catch (error) {
    console.error('Error creating profile:', error);
    res.status(500).json({ error: error.message });
  }
});

app.post('/search', async (req, res) => {
  try {
    const { error, value } = searchSchema.validate(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    const cacheKey = `search:${JSON.stringify(value)}`;
    const cachedResult = await redis.get(cacheKey);
    if (cachedResult) {
      return res.json(JSON.parse(cachedResult));
    }

    const textEmbedding = await getClipEmbeddings('text', value.text_query);

    const searchQuery = {
      vector: textEmbedding,
      topK: value.per_page * 4, 
      includeMetadata: true,
    };

    if (Object.keys(value.filters).length > 0) {
      const filterConditions = {};
      for (const [key, val] of Object.entries(value.filters)) {
        if (val !== undefined && val !== null) {
          filterConditions[key] = val;
        }
      }
      if (Object.keys(filterConditions).length > 0) {
        searchQuery.filter = filterConditions;
      }
    }

    const searchResponse = await profileIndex.query(searchQuery);

    const scores = searchResponse.matches.map(m => m.score);
    const maxScore = Math.max(...scores, 0);
    const meanScore = scores.reduce((a, b) => a + b, 0) / scores.length;
    
    let threshold = Math.max(
      0.12, 
      meanScore * 0.7 
    );

    let results = searchResponse.matches
      .map(match => ({
        profile_id: match.id,
        similarity_score: match.score,
        metadata: match.metadata,
        relative_score: match.score / maxScore 
      }))
      .filter(result => {
        return result.similarity_score >= threshold || 
               result.relative_score >= 0.7;
      })
      .sort((a, b) => b.similarity_score - a.similarity_score)
      .slice(0, value.per_page);

    console.log({
      query: value.text_query,
      stats: {
        maxScore,
        meanScore,
        threshold,
        totalCandidates: searchResponse.matches.length,
        filteredResults: results.length
      },
      allScores: scores.sort((a, b) => b - a).slice(0, 5), 
      matches: results.map(r => ({
        id: r.profile_id,
        score: r.similarity_score.toFixed(4),
        relative_score: r.relative_score.toFixed(4)
      }))
    });

    const response = {
      results: results.map(({ relative_score, ...rest }) => rest),
      meta: {
        total_candidates: searchResponse.matches.length,
        filtered_count: results.length,
        max_score: maxScore,
        mean_score: meanScore,
        threshold: threshold,
        query: value.text_query
      }
    };

    await redis.setex(cacheKey, 300, JSON.stringify(response));
    res.json(response);

  } catch (error) {
    console.error('Error searching profiles:', error);
    res.status(500).json({ 
      error: error.message,
      meta: {
        query: req.body.text_query,
        timestamp: new Date().toISOString()
      }
    });
  }
});

// New route for profile similarities
app.post('/profile-similarities', async (req, res) => {
  try {
    const { error, value } = profileSimilaritySchema.validate(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    const cacheKey = `profile:similarities:${value.profile_id}:${value.target_profile_id}`;
    const cachedResult = await redis.get(cacheKey);
    if (cachedResult) {
      return res.json(JSON.parse(cachedResult));
    }

    // Get both profiles from Pinecone
    const [profile1, profile2] = await Promise.all([
      profileIndex.fetch([value.profile_id]),
      profileIndex.fetch([value.target_profile_id])
    ]);

    if (!profile1.records[value.profile_id] || !profile2.records[value.target_profile_id]) {
      return res.status(404).json({ error: 'One or both profiles not found' });
    }

    const profile1Data = profile1.records[value.profile_id];
    const profile2Data = profile2.records[value.target_profile_id];

    // Compare embeddings to find similarity score
    const similarity = cosineSimilarity(profile1Data.values, profile2Data.values);

    const message = await generateSimilarityMessage(
      profile1Data.metadata,
      profile2Data.metadata,
      similarity
    );

    const response = {
      profiles: {
        profile1: {
          name: profile1Data.metadata.name,
          attributes: {
            age: profile1Data.metadata.age,
            eye_color: profile1Data.metadata.eye_color,
            hair_type: profile1Data.metadata.hair_type,
            ...profile1Data.metadata.additional_attributes
          }
        },
        profile2: {
          name: profile2Data.metadata.name,
          attributes: {
            age: profile2Data.metadata.age,
            eye_color: profile2Data.metadata.eye_color,
            hair_type: profile2Data.metadata.hair_type,
            ...profile2Data.metadata.additional_attributes
          }
        }
      },
      similarity_score: similarity,
      personalized_message: message
    };

    await redis.setex(cacheKey, 300, JSON.stringify(response));
    res.json(response);

  } catch (error) {
    console.error('Error finding profile similarities:', error);
    res.status(500).json({ error: error.message });
  }
});

app.post('/agent-response', async (req, res) => {
  try {
    const { error, value } = agentResponseSchema.validate(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    const responseEmbedding = await getClipEmbeddings('text', value.response);
    
    const recordId = `${value.agentId}-${value.userId}-${Date.now()}`;
    
    const metadata = {
      userId: value.userId,
      agentId: value.agentId,
      question: value.question,
      response: value.response,
      userName: value.userMetadata.name,
      userAge: value.userMetadata.age,
      ...value.userMetadata.additionalInfo,
      timestamp: new Date().toISOString()
    };

    await agentIndex.upsert([{
      id: recordId,
      values: responseEmbedding,
      metadata: metadata
    }]);

    const cachePattern = `agent:search:${value.agentId}:*`;
    const keys = await redis.keys(cachePattern);
    if (keys.length > 0) {
      await redis.del(keys);
    }

    res.status(201).json({
      message: 'Response stored successfully',
      responseId: recordId
    });

  } catch (error) {
    console.error('Error storing agent response:', error);
    res.status(500).json({ error: error.message });
  }
});

app.post('/search-similar-responses', async (req, res) => {
  try {
    const { error, value } = agentSearchSchema.validate(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    const cacheKey = `agent:search:${value.agentId}:${JSON.stringify(value)}`;
    const cachedResult = await redis.get(cacheKey);
    if (cachedResult) {
      return res.json(JSON.parse(cachedResult));
    }

    const responseEmbedding = await getClipEmbeddings('text', value.response);
    console.log(`Searching for similar responses under agent ID: ${value.agentId}`);

    const searchQuery = {
      vector: responseEmbedding,
      topK: value.per_page * 4, 
      includeMetadata: true,
      filter: {
        agentId: { $eq: value.agentId } 
      }
    };

    if (Object.keys(value.filters).length > 0) {
      searchQuery.filter = {
        $and: [
          { agentId: { $eq: value.agentId } },
          value.filters
        ]
      };
    }

    console.log('Executing search with query:', JSON.stringify(searchQuery, null, 2));
    const searchResponse = await agentIndex.query(searchQuery);

    const scores = searchResponse.matches.map(m => m.score);
    const maxScore = Math.max(...scores, 0);
    const meanScore = scores.reduce((a, b) => a + b, 0) / scores.length;
    
    const threshold = Math.max(0.12, meanScore * 0.7);

    const results = searchResponse.matches
      .map(match => ({
        responseId: match.id,
        similarity_score: match.score,
        metadata: {
          userId: match.metadata.userId,
          userName: match.metadata.userName,
          userAge: match.metadata.userAge,
          response: match.metadata.response,
          timestamp: match.metadata.timestamp
        },
        relative_score: match.score / maxScore
      }))
      .filter(result => 
        result.similarity_score >= threshold || 
        result.relative_score >= 0.7
      )
      .sort((a, b) => b.similarity_score - a.similarity_score)
      .slice(0, value.per_page);

    const response = {
      results,
      meta: {
        agentId: value.agentId,
        total_matches: searchResponse.matches.length,
        filtered_count: results.length,
        threshold,
        query_response: value.response
      }
    };

    await redis.setex(cacheKey, 300, JSON.stringify(response));
    res.json(response);

  } catch (error) {
    console.error('Error searching agent responses:', error);
    res.status(500).json({ 
      error: error.message,
      meta: {
        agentId: req.body.agentId,
        timestamp: new Date().toISOString()
      }
    });
  }
});

app.get('/health', async (req, res) => {
  try {
    await redis.ping();
    await Promise.all([
      profileIndex.describeIndexStats(),
      agentIndex.describeIndexStats()
    ]);
    await axios.get(`${CLIP_API_URL}/health`);
    
    res.json({
      status: 'healthy',
      services: {
        api: 'up',
        redis: 'up',
        pinecone: 'up',
        clip_api: 'up'
      }
    });
  } catch (error) {
    res.status(503).json({
      status: 'unhealthy',
      error: error.message
    });
  }
});

app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({
    error: 'Internal server error',
    message: err.message
  });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

module.exports = app;