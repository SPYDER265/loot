import { HfInference } from '@huggingface/inference';

const hf = new HfInference(import.meta.env.VITE_HUGGINGFACE_API_KEY);

export interface AIResponse {
  success: boolean;
  data?: any;
  error?: string;
}

export class HuggingFaceService {
  private static instance: HuggingFaceService;
  private model = 'Qwen/Qwen2.5-VL-7B-Instruct';

  static getInstance(): HuggingFaceService {
    if (!HuggingFaceService.instance) {
      HuggingFaceService.instance = new HuggingFaceService();
    }
    return HuggingFaceService.instance;
  }

  async enhanceOCRText(extractedText: string, imageContext?: string): Promise<AIResponse> {
    try {
      const prompt = `You are an advanced OCR enhancement AI. Your task is to improve the accuracy and readability of extracted text from images.

Original OCR Text:
${extractedText}

${imageContext ? `Image Context: ${imageContext}` : ''}

Please:
1. Correct obvious OCR errors (common character misrecognitions like 0/O, 1/I/l, etc.)
2. Fix spacing and formatting issues
3. Complete truncated words based on context
4. Organize the text into logical structure (tables, lists, paragraphs)
5. Preserve all original information while improving readability

Return only the enhanced text without explanations.`;

      const response = await hf.textGeneration({
        model: this.model,
        inputs: prompt,
        parameters: {
          max_new_tokens: 2000,
          temperature: 0.3,
          top_p: 0.9,
          return_full_text: false
        }
      });

      return {
        success: true,
        data: response.generated_text?.trim() || extractedText
      };
    } catch (error) {
      console.error('Hugging Face OCR enhancement error:', error);
      return {
        success: false,
        error: 'Failed to enhance OCR text',
        data: extractedText
      };
    }
  }

  async cleanAndStructureData(data: any[], filename: string, fileType: string): Promise<AIResponse> {
    try {
      // Sample the data for analysis (first 10 rows to avoid token limits)
      const sampleData = data.slice(0, 10);
      const dataPreview = JSON.stringify(sampleData, null, 2);
      
      const prompt = `You are an expert data cleaning and structuring AI. Analyze this dataset and provide cleaning recommendations.

Dataset Info:
- Filename: ${filename}
- Type: ${fileType}
- Total Rows: ${data.length}
- Sample Data (first 10 rows):
${dataPreview}

Please analyze and provide:
1. Data quality issues identified
2. Recommended cleaning steps
3. Suggested column standardizations
4. Pattern detection (emails, phones, dates, addresses)
5. Data type recommendations for each column

Format your response as JSON with this structure:
{
  "quality_issues": ["issue1", "issue2"],
  "cleaning_recommendations": ["step1", "step2"],
  "column_analysis": {
    "column_name": {
      "type": "text|number|date|email|phone",
      "issues": ["issue1"],
      "suggestions": ["suggestion1"]
    }
  },
  "patterns_detected": {
    "emails": ["column_names"],
    "phones": ["column_names"],
    "dates": ["column_names"],
    "addresses": ["column_names"]
  }
}

Return only valid JSON without explanations.`;

      const response = await hf.textGeneration({
        model: this.model,
        inputs: prompt,
        parameters: {
          max_new_tokens: 1500,
          temperature: 0.2,
          top_p: 0.8,
          return_full_text: false
        }
      });

      let analysisResult;
      try {
        analysisResult = JSON.parse(response.generated_text || '{}');
      } catch {
        // Fallback analysis if JSON parsing fails
        analysisResult = this.generateFallbackAnalysis(data);
      }

      return {
        success: true,
        data: analysisResult
      };
    } catch (error) {
      console.error('Hugging Face data cleaning error:', error);
      return {
        success: false,
        error: 'Failed to analyze data',
        data: this.generateFallbackAnalysis(data)
      };
    }
  }

  async generateChatResponse(userMessage: string, dataContext: any[], filename: string): Promise<AIResponse> {
    try {
      // Create data summary for context
      const dataSummary = this.createDataSummary(dataContext, filename);
      
      const prompt = `You are DataMind AI, an expert data analytics assistant. You help users understand and analyze their data.

Current Dataset Context:
${dataSummary}

User Question: ${userMessage}

Please provide a helpful, insightful response about the user's data. Include:
1. Direct answer to their question
2. Relevant insights from their dataset
3. Actionable recommendations
4. Specific examples from their data when relevant

Keep responses conversational but informative. Use emojis sparingly for visual appeal.`;

      const response = await hf.textGeneration({
        model: this.model,
        inputs: prompt,
        parameters: {
          max_new_tokens: 1000,
          temperature: 0.7,
          top_p: 0.9,
          return_full_text: false
        }
      });

      return {
        success: true,
        data: response.generated_text?.trim() || "I'm here to help analyze your data. Could you please rephrase your question?"
      };
    } catch (error) {
      console.error('Hugging Face chat error:', error);
      return {
        success: false,
        error: 'Failed to generate response',
        data: "I'm experiencing some technical difficulties. Please try again."
      };
    }
  }

  async analyzeImageForOCR(imageFile: File): Promise<AIResponse> {
    try {
      const prompt = `Analyze this image and extract all visible text with high accuracy. Pay special attention to:
1. Tables and structured data
2. Forms and labels
3. Handwritten text
4. Numbers and dates
5. Email addresses and phone numbers

Provide the extracted text in a clean, organized format.`;

      // Convert image to base64 for the vision model
      const imageBase64 = await this.fileToBase64(imageFile);
      
      const response = await hf.visualQuestionAnswering({
        model: this.model,
        inputs: {
          question: prompt,
          image: imageBase64
        }
      });

      return {
        success: true,
        data: response.answer || ''
      };
    } catch (error) {
      console.error('Hugging Face image analysis error:', error);
      return {
        success: false,
        error: 'Failed to analyze image',
        data: ''
      };
    }
  }

  private async fileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result as string;
        resolve(result.split(',')[1]); // Remove data:image/... prefix
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  private createDataSummary(data: any[], filename: string): string {
    if (data.length === 0) return 'No data available';

    const columns = Object.keys(data[0] || {});
    const sampleRows = data.slice(0, 3);
    
    return `Dataset: ${filename}
Rows: ${data.length}
Columns: ${columns.join(', ')}
Sample Data: ${JSON.stringify(sampleRows, null, 2)}`;
  }

  private generateFallbackAnalysis(data: any[]) {
    if (data.length === 0) return {};

    const columns = Object.keys(data[0] || {});
    const analysis = {
      quality_issues: [],
      cleaning_recommendations: ['Remove empty rows', 'Trim whitespace', 'Handle missing values'],
      column_analysis: {},
      patterns_detected: {
        emails: [],
        phones: [],
        dates: [],
        addresses: []
      }
    };

    // Basic pattern detection
    columns.forEach(col => {
      const sample = data.slice(0, 10).map(row => row[col]).filter(val => val);
      
      if (sample.some(val => /\S+@\S+\.\S+/.test(String(val)))) {
        analysis.patterns_detected.emails.push(col);
      }
      if (sample.some(val => /[\+]?[1-9]?[\d\s\-\(\)]{7,15}/.test(String(val)))) {
        analysis.patterns_detected.phones.push(col);
      }
      if (sample.some(val => !isNaN(Date.parse(String(val))))) {
        analysis.patterns_detected.dates.push(col);
      }
    });

    return analysis;
  }
}

export const huggingFaceService = HuggingFaceService.getInstance();