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

const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY, 
});

const index = pc.index('swifey');

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

    try {
      await index.upsert([{
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

    const searchResponse = await index.query(searchQuery);

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

app.get('/health', async (req, res) => {
  try {
    await redis.ping();
    console.log('Redis is up');
    console.log('Pinecone is up');
    try {
        await index.describeIndexStats();
        console.log('Pinecone describeIndexStats is up');
    } catch (describeError) {
        console.error('Error with Pinecone describeIndexStats:', describeError);
        throw describeError; 
    }
    await axios.get(`${CLIP_API_URL}/health`);
    console.log('CLIP API is up');
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