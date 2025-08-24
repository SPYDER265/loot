import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Sparkles, BarChart3, TrendingUp, PieChart } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';

interface Message {
  id: string;
  type: 'user' | 'bot';
  content: string;
  timestamp: Date;
  suggestions?: string[];
}

interface AIChatProps {
  data: any[];
  filename: string;
}

export const AIChat = ({ data, filename }: AIChatProps) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  const suggestedQuestions = [
    "What patterns do you see in this data?",
    "Find any outliers or anomalies",
    "What's the data quality like?",
    "Suggest data cleaning steps",
    "Show me correlation insights",
    "Generate a summary report"
  ];

  useEffect(() => {
    if (data.length > 0 && messages.length === 0) {
      // Initial welcome message
      const welcomeMessage: Message = {
        id: 'welcome',
        type: 'bot',
        content: `Hello! I've analyzed your dataset "${filename}" with ${data.length} rows. I can help you clean, analyze, and gain insights from your data. What would you like to explore?`,
        timestamp: new Date(),
        suggestions: suggestedQuestions.slice(0, 3)
      };
      setMessages([welcomeMessage]);
    }
  }, [data, filename, messages.length]);

  const analyzeData = (question: string) => {
    if (data.length === 0) return "No data available to analyze.";

    const columns = Object.keys(data[0]);
    const numRows = data.length;
    const numCols = columns.length;

    // Enhanced AI analysis with pattern detection
    const detectPatterns = () => {
      const patterns = [];
      
      // Detect email patterns
      const emailColumns = columns.filter(col => {
        const sample = data.slice(0, 10).map(row => row[col]).filter(val => val);
        return sample.some(val => /\S+@\S+\.\S+/.test(String(val)));
      });
      if (emailColumns.length > 0) patterns.push(`📧 Email data found in: ${emailColumns.join(', ')}`);
      
      // Detect phone patterns
      const phoneColumns = columns.filter(col => {
        const sample = data.slice(0, 10).map(row => row[col]).filter(val => val);
        return sample.some(val => /[\+]?[1-9]?[\d\s\-\(\)]{7,15}/.test(String(val)));
      });
      if (phoneColumns.length > 0) patterns.push(`📞 Phone numbers found in: ${phoneColumns.join(', ')}`);
      
      // Detect date patterns
      const dateColumns = columns.filter(col => {
        const sample = data.slice(0, 10).map(row => row[col]).filter(val => val);
        return sample.some(val => !isNaN(Date.parse(String(val))));
      });
      if (dateColumns.length > 0) patterns.push(`📅 Date data found in: ${dateColumns.join(', ')}`);
      
      // Detect location patterns
      const locationColumns = columns.filter(col => {
        const sample = data.slice(0, 10).map(row => row[col]).filter(val => val);
        const locationKeywords = ['street', 'city', 'state', 'country', 'address', 'location', 'zip', 'postal'];
        return locationKeywords.some(keyword => col.toLowerCase().includes(keyword)) ||
               sample.some(val => /\d+\s+\w+\s+(street|st|avenue|ave|road|rd|drive|dr|lane|ln|way|blvd)/i.test(String(val)));
      });
      if (locationColumns.length > 0) patterns.push(`📍 Location data found in: ${locationColumns.join(', ')}`);
      
      return patterns;
    };

    // Enhanced correlation analysis
    const findCorrelations = () => {
      const numericColumns = columns.filter(col => {
        const sample = data.slice(0, 100).map(row => row[col]);
        return sample.every(val => !isNaN(Number(val)) && val !== '');
      });
      
      const correlations = [];
      for (let i = 0; i < numericColumns.length; i++) {
        for (let j = i + 1; j < numericColumns.length; j++) {
          const col1 = numericColumns[i];
          const col2 = numericColumns[j];
          
          // Simple correlation check (positive trend)
          const values1 = data.map(row => Number(row[col1])).filter(v => !isNaN(v));
          const values2 = data.map(row => Number(row[col2])).filter(v => !isNaN(v));
          
          if (values1.length > 5 && values2.length > 5) {
            const avg1 = values1.reduce((a, b) => a + b, 0) / values1.length;
            const avg2 = values2.reduce((a, b) => a + b, 0) / values2.length;
            
            const above1 = values1.filter(v => v > avg1).length;
            const above2 = values2.filter(v => v > avg2).length;
            
            if (Math.abs(above1 - above2) < values1.length * 0.2) {
              correlations.push(`${col1} ↔ ${col2}`);
            }
          }
        }
      }
      
      return correlations;
    };
    // Simple analysis based on question
    if (question.toLowerCase().includes('pattern') || question.toLowerCase().includes('insight') || question.toLowerCase().includes('find')) {
      const numericColumns = columns.filter(col => {
        const sample = data.slice(0, 10).map(row => row[col]);
        return sample.every(val => !isNaN(Number(val)) && val !== '');
      });

      const patterns = detectPatterns();
      const correlations = findCorrelations();

      return `I found several patterns in your data:

📊 **Dataset Overview:**
- ${numRows} rows across ${numCols} columns
- ${numericColumns.length} numeric columns detected: ${numericColumns.slice(0, 3).join(', ')}${numericColumns.length > 3 ? '...' : ''}

🔍 **Key Insights:**
${patterns.length > 0 ? patterns.map(p => `- ${p}`).join('\n') : '- No specific data patterns detected'}
${correlations.length > 0 ? `- Potential correlations: ${correlations.slice(0, 3).join(', ')}` : '- No strong correlations found'}
- Data completeness varies across columns

🎯 **Smart Recommendations:**
1. Use the Smart Organization feature to automatically group emails, phones, and locations
2. ${patterns.length > 0 ? 'Clean and standardize the detected pattern data' : 'Focus on data quality assessment'}
3. ${correlations.length > 0 ? 'Explore the correlations I found for deeper insights' : 'Look for hidden relationships in your data'}
💡 **Recommendations:**
1. Start with data quality assessment
2. Explore distributions of key variables
3. Look for correlations between numeric fields`;
    }

    // Enhanced sentence completion and context awareness
    if (question.toLowerCase().includes('help') || question.toLowerCase().includes('what') || question.toLowerCase().includes('how')) {
      const suggestions = [
        "analyze patterns in my data",
        "find correlations between columns", 
        "identify data quality issues",
        "suggest data cleaning steps",
        "group similar data together",
        "find outliers and anomalies",
        "create meaningful visualizations",
        "extract insights from the data"
      ];
      
      return `🤖 **I can help you:**

${suggestions.map(s => `• ${s.charAt(0).toUpperCase() + s.slice(1)}`).join('\n')}

**Try asking me:**
- "What patterns do you see in this data?"
- "Find correlations in my dataset"
- "Help me clean this data"
- "What insights can you provide?"

I'm designed to understand your data and provide intelligent recommendations!`;
    }
    if (question.toLowerCase().includes('outlier') || question.toLowerCase().includes('anomal')) {
      const numericColumns = columns.filter(col => {
        const sample = data.slice(0, 10).map(row => row[col]);
        return sample.every(val => !isNaN(Number(val)) && val !== '');
      });
      
      const outlierAnalysis = numericColumns.slice(0, 3).map(col => {
        const values = data.map(row => Number(row[col])).filter(v => !isNaN(v));
        const sorted = values.sort((a, b) => a - b);
        const q1 = sorted[Math.floor(sorted.length * 0.25)];
        const q3 = sorted[Math.floor(sorted.length * 0.75)];
        const iqr = q3 - q1;
        const lowerBound = q1 - 1.5 * iqr;
        const upperBound = q3 + 1.5 * iqr;
        const outliers = values.filter(v => v < lowerBound || v > upperBound);
        
        return `**${col}**: ${outliers.length} potential outliers (${((outliers.length / values.length) * 100).toFixed(1)}%)`;
      });

      return `🔍 **Outlier Analysis:**

${outlierAnalysis.length > 0 ? outlierAnalysis.join('\n') : 'No numeric columns available for outlier analysis.'}

**Detection Method**: Using IQR (Interquartile Range) method
- Values below Q1 - 1.5×IQR or above Q3 + 1.5×IQR are flagged
- This is a statistical approach to identify unusual values

**Next Steps:**
1. Review flagged values to determine if they're errors or valid extreme values
2. Consider data transformation or removal based on business context
3. Use visualization to better understand the distribution

💡 **Tip**: Not all outliers are errors - some might be valuable insights!`;
    }

    if (question.toLowerCase().includes('quality') || question.toLowerCase().includes('clean')) {
      const missingData = columns.map(col => {
        const missing = data.filter(row => !row[col] || row[col] === '').length;
        return { column: col, missing, percentage: ((missing / numRows) * 100).toFixed(1) };
      }).filter(item => item.missing > 0);

      return `📋 **Data Quality Report:**

**Completeness Analysis:**
${missingData.length === 0 ? 
  '✅ No missing values detected!' : 
  missingData.map(item => `- ${item.column}: ${item.missing} missing (${item.percentage}%)`).join('\n')
}

**Quality Recommendations:**
1. ${missingData.length > 0 ? 'Handle missing values through imputation or removal' : 'Data appears complete'}
2. Check for duplicate rows
3. Validate data types and formats
4. Standardize text fields (case, spacing)
5. Remove or flag obvious errors

Would you like me to suggest specific cleaning steps for any column?`;
    }

    // Default response
    return `I understand you're asking: "${question}"

Based on your dataset "${filename}", I can help with:

🔍 **Analysis Available:**
- Statistical summaries
- Data quality assessment  
- Pattern recognition
- Outlier detection
- Correlation analysis

📊 **Current Dataset:**
- ${numRows} rows
- ${numCols} columns
- File type: ${filename.split('.').pop()?.toUpperCase()}

What specific aspect would you like to explore further?`;
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    // Simulate AI processing
    setTimeout(() => {
      const response = analyzeData(input);
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: response,
        timestamp: new Date(),
        suggestions: suggestedQuestions.filter(q => !q.toLowerCase().includes(input.toLowerCase().split(' ')[0])).slice(0, 2)
      };

      setMessages(prev => [...prev, botMessage]);
      setIsLoading(false);
    }, 1500);
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInput(suggestion);
  };

  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <Card className="glass-card h-[450px] sm:h-[600px] flex flex-col animate-fade-in">
      <div className="p-3 sm:p-4 border-b border-border/50">
        <div className="flex items-center gap-2 sm:gap-3">
          <div className="p-2 rounded-full bg-gradient-to-r from-chart-primary to-chart-secondary">
            <Bot className="h-5 w-5 text-primary-foreground" />
          </div>
          <div>
            <h3 className="text-sm sm:text-base font-semibold">AI Analytics Assistant</h3>
            <p className="text-xs sm:text-sm text-muted-foreground">Ask me anything about your data</p>
          </div>
          <div className="ml-auto flex gap-1 hidden sm:flex">
            <Badge variant="secondary">
              <Sparkles className="h-3 w-3 mr-1" />
              Smart Analysis
            </Badge>
          </div>
        </div>
      </div>

      <ScrollArea className="flex-1 p-3 sm:p-4" ref={scrollAreaRef}>
        <div className="space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex gap-2 sm:gap-3 ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {message.type === 'bot' && (
                <div className="p-2 rounded-full bg-gradient-to-r from-chart-primary to-chart-secondary mt-1">
                  <Bot className="h-4 w-4 text-primary-foreground" />
                </div>
              )}
              
              <div className={`max-w-[85%] sm:max-w-[80%] ${message.type === 'user' ? 'order-first' : ''}`}>
                <div
                  className={`p-2 sm:p-3 rounded-lg text-sm sm:text-base ${
                    message.type === 'user'
                      ? 'bg-primary text-primary-foreground ml-auto'
                      : 'glass-card'
                  }`}
                >
                  <p className="whitespace-pre-line">{message.content}</p>
                </div>
                
                {message.suggestions && (
                  <div className="mt-2 flex flex-wrap gap-1 sm:gap-2 max-w-full">
                    {message.suggestions.map((suggestion, index) => (
                      <Button
                        key={index}
                        variant="outline"
                        size="sm"
                        onClick={() => handleSuggestionClick(suggestion)}
                        className="text-xs whitespace-nowrap"
                      >
                        {suggestion}
                      </Button>
                    ))}
                  </div>
                )}
                
                <p className="text-xs text-muted-foreground mt-1 hidden sm:block">
                  {message.timestamp.toLocaleTimeString()}
                </p>
              </div>

              {message.type === 'user' && (
                <div className="p-2 rounded-full bg-secondary mt-1">
                  <User className="h-4 w-4" />
                </div>
              )}
            </div>
          ))}
          
          {isLoading && (
            <div className="flex gap-2 sm:gap-3">
              <div className="p-2 rounded-full bg-gradient-to-r from-chart-primary to-chart-secondary mt-1">
                <Bot className="h-4 w-4 text-primary-foreground" />
              </div>
              <div className="glass-card p-3 rounded-lg">
                <div className="flex items-center gap-2">
                  <div className="animate-spin h-4 w-4 border-2 border-chart-primary border-t-transparent rounded-full"></div>
                  <span>Analyzing your data...</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </ScrollArea>

      <div className="p-3 sm:p-4 border-t border-border/50">
        <div className="flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Ask about your data..."
            disabled={isLoading}
            className="flex-1"
          />
          <Button
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            variant="analytics"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
        
        <div className="flex gap-1 sm:gap-2 mt-2 sm:mt-3 flex-wrap justify-center sm:justify-start">
          <Button variant="glass" size="sm" onClick={() => handleSuggestionClick("Summarize this dataset")}>
            <BarChart3 className="h-3 w-3 mr-1" />
            <span className="hidden sm:inline">Summary</span>
            <span className="sm:hidden text-xs">Sum</span>
          </Button>
          <Button variant="glass" size="sm" onClick={() => handleSuggestionClick("Check data quality")}>
            <TrendingUp className="h-3 w-3 mr-1" />
            <span className="hidden sm:inline">Quality Check</span>
            <span className="sm:hidden text-xs">Quality</span>
          </Button>
          <Button variant="glass" size="sm" onClick={() => handleSuggestionClick("Find correlations")}>
            <PieChart className="h-3 w-3 mr-1" />
            <span className="hidden sm:inline">Correlations</span>
            <span className="sm:hidden text-xs">Corr</span>
          </Button>
        </div>
      </div>
    </Card>
  );
};