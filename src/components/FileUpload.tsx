import { useState, useCallback } from 'react';
import { Upload, FileText, Image, BarChart3 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { useToast } from '@/hooks/use-toast';
import * as XLSX from 'xlsx';
import Papa from 'papaparse';

interface FileUploadProps {
  onDataLoad: (data: any[], filename: string, type: string) => void;
  onImageLoad: (file: File) => void;
}

export const FileUpload = ({ onDataLoad, onImageLoad }: FileUploadProps) => {
  const [dragActive, setDragActive] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const { toast } = useToast();

  const handleFile = useCallback(async (file: File) => {
    setIsProcessing(true);
    
    try {
      const fileType = file.type;
      const fileName = file.name.toLowerCase();
      
      // Handle images
      if (fileType.startsWith('image/') || fileName.match(/\.(jpg|jpeg|png|gif|bmp|webp)$/)) {
        onImageLoad(file);
        toast({
          title: "Image uploaded successfully",
          description: "Processing image for OCR analysis..."
        });
        setIsProcessing(false);
        return;
      }
      
      // Handle Excel files
      if (fileName.match(/\.(xlsx|xls)$/)) {
        const buffer = await file.arrayBuffer();
        const workbook = XLSX.read(buffer);
        const sheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[sheetName];
        const data = XLSX.utils.sheet_to_json(worksheet);
        
        onDataLoad(data, file.name, 'excel');
        toast({
          title: "Excel file processed",
          description: `Loaded ${data.length} rows from ${file.name}`
        });
        setIsProcessing(false);
        return;
      }
      
      // Handle CSV files
      if (fileName.match(/\.csv$/)) {
        const text = await file.text();
        Papa.parse(text, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {
            onDataLoad(results.data, file.name, 'csv');
            toast({
              title: "CSV file processed",
              description: `Loaded ${results.data.length} rows from ${file.name}`
            });
            setIsProcessing(false);
          },
          error: (error) => {
            setIsProcessing(false);
            throw new Error(`CSV parsing error: ${error.message}`);
          }
        });
        return;
      }
      
      // Handle JSON files
      if (fileName.match(/\.json$/)) {
        const text = await file.text();
        const data = JSON.parse(text);
        const arrayData = Array.isArray(data) ? data : [data];
        
        onDataLoad(arrayData, file.name, 'json');
        toast({
          title: "JSON file processed",
          description: `Loaded ${arrayData.length} records from ${file.name}`
        });
        setIsProcessing(false);
        return;
      }
      
      throw new Error('Unsupported file type. Please upload CSV, Excel, JSON, or image files.');
      
    } catch (error) {
      console.error('File processing error:', error);
      toast({
        title: "Error processing file",
        description: error instanceof Error ? error.message : "Unknown error occurred",
        variant: "destructive"
      });
      setIsProcessing(false);
    }
  }, [onDataLoad, onImageLoad, toast]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFile(files[0]);
    }
  }, [handleFile]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  }, [handleFile]);

  return (
    <div className="p-3 sm:p-8">
      <div className="text-center mb-8">
        <h2 className="text-2xl sm:text-3xl font-bold mb-4 bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
          Upload Your Data for AI Analysis
        </h2>
        <p className="text-sm sm:text-base text-muted-foreground max-w-2xl mx-auto px-2 sm:px-4">
          Drag and drop your files or click to browse. Powered by Hugging Face AI for intelligent analysis of CSV, Excel, JSON, and images with enhanced OCR.
        </p>
      </div>
      
      <div
        className={`relative border-2 border-dashed rounded-2xl p-6 sm:p-16 text-center transition-all duration-500 overflow-hidden ${
          dragActive 
            ? 'border-primary bg-gradient-to-br from-primary/20 to-accent/10 scale-105 glow-effect' 
            : 'border-border/30 hover:border-primary/50 hover:bg-gradient-to-br hover:from-primary/5 hover:to-accent/5'
        }`}
        onDragEnter={() => setDragActive(true)}
        onDragLeave={() => setDragActive(false)}
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
      >
        {/* Background decoration */}
        <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-accent/5 opacity-50" />
        
        <div className="relative flex flex-col items-center gap-4 sm:gap-8">
          <div className={`relative p-4 sm:p-6 rounded-full bg-gradient-to-r from-chart-primary to-chart-secondary transition-transform duration-300 ${dragActive ? 'scale-110 animate-bounce-gentle' : 'hover:scale-105'}`}>
            <Upload className="h-12 w-12 text-primary-foreground" />
            {dragActive && (
              <div className="absolute inset-0 rounded-full bg-gradient-to-r from-chart-primary to-chart-secondary animate-ping opacity-75" />
            )}
          </div>
          
          <div className="space-y-4">
            <h3 className="text-xl sm:text-2xl font-semibold">
              {dragActive ? 'Release for AI analysis!' : 'Drop your data files for AI analysis'}
            </h3>
            <p className="text-muted-foreground text-sm sm:text-lg px-2">
              {dragActive 
                ? 'Let go to start AI processing your file' 
                : 'Support for CSV, Excel, JSON, and image files with AI-enhanced OCR'
              }
            </p>
          </div>
          
          <div className="grid grid-cols-3 gap-2 sm:gap-6 max-w-lg w-full">
            <div className="flex flex-col items-center gap-2 p-4 rounded-xl bg-gradient-to-br from-primary/10 to-primary/5 border border-primary/20">
              <FileText className="h-8 w-8 text-primary" />
              <span className="text-xs sm:text-sm font-medium">CSV/Excel</span>
            </div>
            <div className="flex flex-col items-center gap-2 p-4 rounded-xl bg-gradient-to-br from-accent/10 to-accent/5 border border-accent/20">
              <BarChart3 className="h-8 w-8 text-accent" />
              <span className="text-xs sm:text-sm font-medium">JSON</span>
            </div>
            <div className="flex flex-col items-center gap-2 p-4 rounded-xl bg-gradient-to-br from-chart-secondary/10 to-chart-secondary/5 border border-chart-secondary/20">
              <Image className="h-8 w-8 text-chart-secondary" />
              <span className="text-xs sm:text-sm font-medium">AI OCR</span>
            </div>
          </div>
          
          <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 w-full max-w-md">
            <Button 
              variant="analytics" 
              size="default" 
              disabled={isProcessing}
              className={`transition-all duration-300 w-full sm:w-auto h-11 ${isProcessing ? 'animate-pulse' : 'hover:scale-105'}`}
              onClick={() => document.getElementById('file-upload')?.click()}
            >
              {isProcessing ? (
                <>
                  <div className="animate-spin mr-2 h-5 w-5 border-2 border-primary-foreground border-t-transparent rounded-full" />
                  AI Processing...
                </>
              ) : (
                <>
                  <span className="mr-2">📁</span>
                  Browse Files
                </>
              )}
            </Button>
            <Button variant="glass" size="default" className="hover:scale-105 transition-transform w-full sm:w-auto h-11">
              <span className="mr-2">👁️</span>
              AI Samples
            </Button>
          </div>
          
          <input
            id="file-upload"
            type="file"
            className="hidden"
            accept=".csv,.xlsx,.xls,.json,.jpg,.jpeg,.png,.gif,.bmp,.webp"
            onChange={handleFileInput}
          />
        </div>
      </div>
    </div>
  );
};