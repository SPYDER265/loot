import { useNavigate } from 'react-router-dom';
import { Brain, Database, Image, TrendingUp, Zap, Shield, ChartColumn, FileText, Upload } from 'lucide-react';
import { Button } from '@/components/ui/button';

const Landing = () => {
  const navigate = useNavigate();

  const handleStartAnalyzing = () => {
    navigate('/app');
  };

  const handleTryAIAssistant = () => {
    navigate('/app');
  };

  const handleUploadFiles = () => {
    navigate('/app');
  };

  const handleUploadImages = () => {
    navigate('/app');
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0">
          <img 
            src="https://images.pexels.com/photos/590022/pexels-photo-590022.jpeg?auto=compress&cs=tinysrgb&w=1920&h=1080&fit=crop" 
            alt="Data Analytics Dashboard" 
            className="w-full h-full object-cover"
          />
          <div className="absolute inset-0 bg-background/80 backdrop-blur-sm"></div>
        </div>
        
        <div className="relative container mx-auto px-4 py-24">
          <div className="max-w-4xl mx-auto text-center space-y-8">
            <div className="space-y-4">
              <h1 className="text-4xl md:text-6xl font-bold bg-gradient-to-r from-chart-primary to-chart-secondary bg-clip-text text-transparent">
                Hugging Face AI-Powered Analytics
              </h1>
              <p className="text-xl md:text-2xl text-muted-foreground max-w-2xl mx-auto">
                Transform your raw data into actionable insights with Hugging Face's advanced AI models
              </p>
            </div>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button 
                onClick={handleStartAnalyzing}
                className="bg-gradient-to-r from-chart-primary to-chart-secondary hover:opacity-90 text-primary-foreground px-8 h-11"
              >
                <Zap className="w-5 h-5 mr-2" />
                Start Analyzing
              </Button>
              <Button 
                variant="outline" 
                onClick={handleTryAIAssistant}
                className="glass-card border-border/50 backdrop-blur-md px-8 h-11"
              >
                <Brain className="w-5 h-5 mr-2" />
                Try AI Assistant
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 relative">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
              Powerful Features for Data Excellence
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Everything you need to clean, analyze, and visualize your data with AI assistance
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-16">
            <div className="glass-card p-6 hover:border-chart-primary/50 transition-all duration-300 floating-card">
              <div className="w-12 h-12 bg-chart-primary rounded-lg flex items-center justify-center mb-4">
                <Database className="w-6 h-6 text-primary-foreground" />
              </div>
              <h3 className="text-xl font-semibold text-foreground mb-2">Smart Data Cleaning</h3>
              <p className="text-muted-foreground">
                Automatically detect and fix data quality issues with Hugging Face AI-powered cleaning algorithms
              </p>
            </div>

            <div className="glass-card p-6 hover:border-chart-secondary/50 transition-all duration-300 floating-card">
              <div className="w-12 h-12 bg-chart-secondary rounded-lg flex items-center justify-center mb-4">
                <Image className="w-6 h-6 text-primary-foreground" />
              </div>
              <h3 className="text-xl font-semibold text-foreground mb-2">OCR Text Extraction</h3>
              <p className="text-muted-foreground">
                Convert images and PDFs to analyzable text data with Hugging Face AI-enhanced OCR technology
              </p>
            </div>

            <div className="glass-card p-6 hover:border-chart-tertiary/50 transition-all duration-300 floating-card">
              <div className="w-12 h-12 bg-chart-tertiary rounded-lg flex items-center justify-center mb-4">
                <TrendingUp className="w-6 h-6 text-primary-foreground" />
              </div>
              <h3 className="text-xl font-semibold text-foreground mb-2">Interactive Visualizations</h3>
              <p className="text-muted-foreground">
                Create stunning charts and dashboards that reveal hidden patterns in your data
              </p>
            </div>

            <div className="glass-card p-6 hover:border-chart-primary/50 transition-all duration-300 floating-card">
              <div className="w-12 h-12 bg-chart-primary rounded-lg flex items-center justify-center mb-4">
                <Brain className="w-6 h-6 text-primary-foreground" />
              </div>
              <h3 className="text-xl font-semibold text-foreground mb-2">AI Recommendations</h3>
              <p className="text-muted-foreground">
                Get intelligent suggestions powered by Hugging Face AI for data analysis and visualization strategies
              </p>
            </div>

            <div className="glass-card p-6 hover:border-chart-secondary/50 transition-all duration-300 floating-card">
              <div className="w-12 h-12 bg-chart-secondary rounded-lg flex items-center justify-center mb-4">
                <Zap className="w-6 h-6 text-primary-foreground" />
              </div>
              <h3 className="text-xl font-semibold text-foreground mb-2">Real-time Processing</h3>
              <p className="text-muted-foreground">
                Process large datasets quickly with optimized algorithms and real-time feedback
              </p>
            </div>

            <div className="glass-card p-6 hover:border-chart-tertiary/50 transition-all duration-300 floating-card">
              <div className="w-12 h-12 bg-chart-tertiary rounded-lg flex items-center justify-center mb-4">
                <Shield className="w-6 h-6 text-primary-foreground" />
              </div>
              <h3 className="text-xl font-semibold text-foreground mb-2">Secure & Private</h3>
              <p className="text-muted-foreground">
                Your data stays secure with enterprise-grade encryption and privacy controls
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Upload Section */}
      <section className="py-20 bg-gradient-to-br from-background to-muted/20">
        <div className="container mx-auto px-4">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
              Get Started with Your Data
            </h2>
            <p className="text-lg text-muted-foreground">
              Upload your datasets or images and let our AI assistant guide you through the analysis
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="glass-card relative overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-br from-chart-primary/10 to-chart-secondary/10"></div>
              <div className="relative p-8 border-2 border-dashed rounded-lg transition-all duration-300 border-muted-foreground/30">
                <div className="text-center space-y-4">
                  <div className="mx-auto w-16 h-16 bg-gradient-to-r from-chart-primary to-chart-secondary rounded-full flex items-center justify-center">
                    <ChartColumn className="w-8 h-8 text-primary-foreground" />
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-foreground mb-2">Upload Dataset</h3>
                    <p className="text-muted-foreground mb-4">Drop your CSV, Excel, or JSON files here</p>
                  </div>
                  <div className="space-y-2">
                    <Button 
                      onClick={handleUploadFiles}
                      variant="secondary" 
                      className="w-full"
                    >
                      <Upload className="w-4 h-4 mr-2" />
                      Choose Files
                    </Button>
                    <p className="text-xs text-muted-foreground">
                      Supports CSV, Excel, JSON files up to 50MB
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="glass-card relative overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-br from-chart-secondary/10 to-chart-tertiary/10"></div>
              <div className="relative p-8 border-2 border-dashed rounded-lg transition-all duration-300 border-muted-foreground/30">
                <div className="text-center space-y-4">
                  <div className="mx-auto w-16 h-16 bg-chart-secondary rounded-full flex items-center justify-center">
                    <Image className="w-8 h-8 text-primary-foreground" />
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-foreground mb-2">Extract Text from Images</h3>
                    <p className="text-muted-foreground mb-4">Upload images to extract and analyze text data</p>
                  </div>
                  <div className="space-y-2">
                    <Button 
                      onClick={handleUploadImages}
                      variant="secondary" 
                      className="w-full"
                    >
                      <FileText className="w-4 h-4 mr-2" />
                      Upload Images
                    </Button>
                    <p className="text-xs text-muted-foreground">
                      PNG, JPG, PDF supported
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Landing;