import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Sparkles, BarChart3, TrendingUp, PieChart } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { huggingFaceService } from '@/lib/huggingface';

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
  const [isAIThinking, setIsAIThinking] = useState(false);
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
    setIsAIThinking(true);

    try {
      // Use Hugging Face AI for intelligent responses
      const aiResponse = await huggingFaceService.generateChatResponse(input, data, filename);
      
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: aiResponse.success ? aiResponse.data : "I'm having trouble processing your request. Please try again.",
        timestamp: new Date(),
        suggestions: suggestedQuestions.filter(q => !q.toLowerCase().includes(input.toLowerCase().split(' ')[0])).slice(0, 2)
      };

      setMessages(prev => [...prev, botMessage]);
      
      if (aiResponse.success) {
        toast({
          title: "AI Response Generated",
          description: "Powered by Hugging Face Qwen model"
        });
      }
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: "I'm experiencing technical difficulties. Please try your question again.",
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setIsAIThinking(false);
    }
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
            <p className="text-xs sm:text-sm text-muted-foreground">
              Powered by Hugging Face AI • Ask me anything about your data
            </p>
          </div>
          <div className="ml-auto flex gap-1 hidden sm:flex">
            <Badge variant="secondary">
              <Sparkles className="h-3 w-3 mr-1" />
              HF AI
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
          
          {(isLoading || isAIThinking) && (
            <div className="flex gap-2 sm:gap-3">
              <div className="p-2 rounded-full bg-gradient-to-r from-chart-primary to-chart-secondary mt-1">
                <Bot className="h-4 w-4 text-primary-foreground" />
              </div>
              <div className="glass-card p-3 rounded-lg">
                <div className="flex items-center gap-2">
                  <div className="animate-spin h-4 w-4 border-2 border-chart-primary border-t-transparent rounded-full"></div>
                  <span>{isAIThinking ? 'AI thinking...' : 'Analyzing your data...'}</span>
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
            placeholder="Ask AI about your data..."
            disabled={isLoading || isAIThinking}
            className="flex-1"
          />
          <Button
            onClick={handleSend}
            disabled={!input.trim() || isLoading || isAIThinking}
            variant="analytics"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
        
        <div className="flex gap-1 sm:gap-2 mt-2 sm:mt-3 flex-wrap justify-center sm:justify-start">
          <Button variant="glass" size="sm" onClick={() => handleSuggestionClick("Summarize this dataset")}>
            <BarChart3 className="h-3 w-3 mr-1" />
            <span className="hidden sm:inline">AI Summary</span>
            <span className="sm:hidden text-xs">AI Sum</span>
          </Button>
          <Button variant="glass" size="sm" onClick={() => handleSuggestionClick("Check data quality")}>
            <TrendingUp className="h-3 w-3 mr-1" />
            <span className="hidden sm:inline">AI Quality</span>
            <span className="sm:hidden text-xs">Quality</span>
          </Button>
          <Button variant="glass" size="sm" onClick={() => handleSuggestionClick("Find correlations")}>
            <PieChart className="h-3 w-3 mr-1" />
            <span className="hidden sm:inline">AI Insights</span>
            <span className="sm:hidden text-xs">Insights</span>
          </Button>
        </div>
      </div>
    </Card>
  );
};