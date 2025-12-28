# API Cost Analysis & Optimization

## Current API Usage Per Audio File

### API Calls Made:
1. **Transcription**: 1 call to `claude-3-5-sonnet-20241022`
   - Model: Claude 3.5 Sonnet ($3/$15 per million tokens)
   - Max tokens: 4096
   - Includes audio data (base64 encoded)

2. **Claim Extraction**: 1 call to `claude-3-5-sonnet-20241022`
   - Model: Claude 3.5 Sonnet ($3/$15 per million tokens)
   - Max tokens: 2048

3. **Fact-Checking**: N calls (one per claim)
   - Model: Claude 3.5 Sonnet ($3/$15 per million tokens)
   - Max tokens: 4096
   - **CACHED** - Repeated claims don't cost extra
   - Potential additional calls if tool_use triggers (up to 5 iterations)

### Estimated Cost Per Request:
- **10-second audio** (~880KB WAV file):
  - Transcription: ~$0.01-0.03 (depends on audio size)
  - Claim extraction: ~$0.005-0.01
  - Fact-checking: ~$0.01-0.03 per claim (cached after first time)
  
- **Total for 3 claims (first time)**: ~$0.05-0.10
- **Total for 3 claims (cached)**: ~$0.02-0.04 (only transcription + extraction)

## Cost Optimization Recommendations

### ✅ Already Implemented:
1. **Caching for fact-checks** - Identical claims are cached
2. **Parallel processing** - Doesn't reduce cost, but improves speed

### 🔧 Recommended Optimizations:

1. **Use Claude Haiku for transcription/extraction** (10x cheaper)
   - Haiku: $0.80/$4 per million tokens
   - Sonnet: $3/$15 per million tokens
   - **Savings: ~75-80%** on transcription and extraction

2. **Add transcription caching** (same audio = same transcript)
   - Hash audio file and cache transcript
   - **Savings: 100%** on repeated audio files

3. **Reduce max_tokens** where appropriate
   - Claim extraction: 2048 → 1024 (sufficient)
   - **Savings: ~20-30%** on token costs

4. **Simplify tool_use handling** (currently adds cost without real web search)
   - Remove unnecessary iterations
   - **Savings: Variable, but can prevent extra API calls**

## Cost Comparison

### Current Setup (Sonnet for everything):
- 10 requests with 3 claims each: ~$0.50-1.00
- 100 requests: ~$5.00-10.00

### Optimized Setup (Haiku for transcription/extraction):
- 10 requests with 3 claims each: ~$0.15-0.30
- 100 requests: ~$1.50-3.00
- **Savings: ~70%**

### With Full Caching:
- 10 unique requests: ~$0.15-0.30
- 10 repeated requests: ~$0.00 (all cached)
- **Savings: ~100% on repeats**

## Recommendations

1. **For development/testing**: Use current setup (acceptable cost)
2. **For production**: Implement Haiku for transcription/extraction
3. **For high volume**: Add transcription caching

