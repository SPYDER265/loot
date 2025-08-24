import { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';
import { Badge } from '@/components/ui/badge';
import { useToast } from '@/components/ui/use-toast';
import { Search, Filter, Download, TrendingUp, ChevronDown } from 'lucide-react';
import * as XLSX from 'xlsx';
import jsPDF from 'jspdf';
import { saveAs } from 'file-saver';

interface DataTableProps {
  data: any[];
  filename: string;
  type: string;
}

export const DataTable = ({ data, filename, type }: DataTableProps) => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [searchTerm, setSearchTerm] = useState('');
  const [sortColumn, setSortColumn] = useState('');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  const [currentPage, setCurrentPage] = useState(1);
  const [rowsPerPage] = useState(10);

  const columns = useMemo(() => {
    if (data.length === 0) return [];
    return Object.keys(data[0]).filter(key => key && key.trim() !== '');
  }, [data]);

  const filteredData = useMemo(() => {
    return data.filter(row =>
      Object.values(row).some(value =>
        String(value).toLowerCase().includes(searchTerm.toLowerCase())
      )
    );
  }, [data, searchTerm]);

  const sortedData = useMemo(() => {
    if (!sortColumn) return filteredData;
    
    return [...filteredData].sort((a, b) => {
      const aVal = a[sortColumn];
      const bVal = b[sortColumn];
      
      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
      }
      
      const aStr = String(aVal).toLowerCase();
      const bStr = String(bVal).toLowerCase();
      
      if (sortDirection === 'asc') {
        return aStr.localeCompare(bStr);
      } else {
        return bStr.localeCompare(aStr);
      }
    });
  }, [filteredData, sortColumn, sortDirection]);

  const paginatedData = useMemo(() => {
    const startIndex = (currentPage - 1) * rowsPerPage;
    return sortedData.slice(startIndex, startIndex + rowsPerPage);
  }, [sortedData, currentPage, rowsPerPage]);

  const totalPages = Math.ceil(sortedData.length / rowsPerPage);

  const getColumnType = (column: string) => {
    const sample = data.slice(0, 100);
    const values = sample.map(row => row[column]).filter(v => v != null);
    
    if (values.every(v => !isNaN(Number(v)))) return 'number';
    if (values.every(v => !isNaN(Date.parse(v)))) return 'date';
    return 'text';
  };

  const getColumnStats = (column: string) => {
    const values = data.map(row => row[column]).filter(v => v != null);
    const unique = new Set(values).size;
    const filled = values.length;
    const total = data.length;
    
    return {
      unique,
      filled,
      total,
      completion: Math.round((filled / total) * 100)
    };
  };

  const handleExportCSV = () => {
    if (data.length === 0) return;
    
    // Convert data to CSV
    const headers = Object.keys(data[0]);
    const csvContent = [
      headers.join(','),
      ...data.map(row => 
        headers.map(header => {
          const value = row[header];
          // Escape commas and quotes in values
          if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
            return `"${value.replace(/"/g, '""')}"`;
          }
          return value ?? '';
        }).join(',')
      )
    ].join('\n');
    
    // Create and download file
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `${filename.replace(/\.[^/.]+$/, '')}_export.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    toast({
      title: "Export Complete",
      description: `${data.length} rows exported to CSV file.`,
    });
  };

  const handleExportExcel = () => {
    if (data.length === 0) return;
    
    try {
      const worksheet = XLSX.utils.json_to_sheet(data);
      const workbook = XLSX.utils.book_new();
      XLSX.utils.book_append_sheet(workbook, worksheet, 'Data');
      
      // Auto-size columns
      const cols = Object.keys(data[0]).map(key => ({ wch: Math.max(key.length, 15) }));
      worksheet['!cols'] = cols;
      
      const excelBuffer = XLSX.write(workbook, { bookType: 'xlsx', type: 'array' });
      const blob = new Blob([excelBuffer], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
      saveAs(blob, `${filename.replace(/\.[^/.]+$/, '')}_export.xlsx`);
      
      toast({
        title: "Excel Export Complete",
        description: `${data.length} rows exported to Excel file.`,
      });
    } catch (error) {
      toast({
        title: "Export Failed",
        description: "Failed to export to Excel format.",
        variant: "destructive"
      });
    }
  };

  const handleExportPDF = () => {
    if (data.length === 0) return;
    
    try {
      const headers = Object.keys(data[0]);
      
      // Create PDF
      const pdf = new jsPDF();
      
      // Add title
      pdf.setFontSize(16);
      pdf.text(`Data Export: ${filename}`, 20, 20);
      
      // Add export date
      pdf.setFontSize(10);
      pdf.text(`Exported on: ${new Date().toLocaleDateString()}`, 20, 30);
      
      // Add table headers
      let yPosition = 50;
      pdf.setFontSize(8);
      pdf.setFont(undefined, 'bold');
      
      const columnWidth = 170 / headers.length; // Distribute columns across page width
      headers.forEach((header, index) => {
        pdf.text(String(header), 20 + (index * columnWidth), yPosition);
      });
      
      // Add data rows (limit to prevent performance issues)
      pdf.setFont(undefined, 'normal');
      const maxRows = Math.min(data.length, 100); // Limit to 100 rows for PDF
      
      for (let i = 0; i < maxRows; i++) {
        yPosition += 10;
        
        // Check if we need a new page
        if (yPosition > 280) {
          pdf.addPage();
          yPosition = 20;
        }
        
        const row = data[i];
        headers.forEach((header, index) => {
          const cellValue = String(row[header] || '');
          // Truncate long values to fit in cell
          const truncatedValue = cellValue.length > 15 ? cellValue.substring(0, 12) + '...' : cellValue;
          pdf.text(truncatedValue, 20 + (index * columnWidth), yPosition);
        });
      }
      
      // Save the PDF
      pdf.save(`${filename.replace(/\.[^/.]+$/, '')}_export.pdf`);
      
      toast({
        title: "PDF Export Complete",
        description: `${Math.min(data.length, 100)} rows exported to PDF file.`,
      });
    } catch (error) {
      toast({
        title: "Export Failed",
        description: "Failed to export to PDF format.",
        variant: "destructive"
      });
    }
  };

  const handleExportTXT = () => {
    if (data.length === 0) return;
    
    try {
      const headers = Object.keys(data[0]);
      let txtContent = `Data Export: ${filename}\n`;
      txtContent += `Exported on: ${new Date().toLocaleDateString()}\n`;
      txtContent += `Total rows: ${data.length}\n\n`;
      txtContent += '='.repeat(50) + '\n\n';
      
      // Add headers
      txtContent += headers.join('\t') + '\n';
      txtContent += '-'.repeat(headers.join('\t').length) + '\n';
      
      // Add data rows
      data.forEach(row => {
        const rowData = headers.map(header => String(row[header] || '')).join('\t');
        txtContent += rowData + '\n';
      });
      
      const blob = new Blob([txtContent], { type: 'text/plain;charset=utf-8' });
      saveAs(blob, `${filename.replace(/\.[^/.]+$/, '')}_export.txt`);
      
      toast({
        title: "TXT Export Complete",
        description: `${data.length} rows exported to text file.`,
      });
    } catch (error) {
      toast({
        title: "Export Failed",
        description: "Failed to export to text format.",
        variant: "destructive"
      });
    }
  };

  if (data.length === 0) {
    return (
      <Card className="glass-card p-8 text-center">
        <p className="text-muted-foreground">No data to display</p>
      </Card>
    );
  }

  return (
    <Card className="glass-card p-3 sm:p-6 space-y-4 sm:space-y-6 animate-fade-in">
      <div className="flex flex-col sm:flex-row items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <h3 className="text-base sm:text-xl font-semibold flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-chart-primary" />
            <span className="truncate">{filename}</span>
          </h3>
          <p className="text-xs sm:text-sm text-muted-foreground">
            {data.length} rows, {columns.length} columns • Type: {type.toUpperCase()}
          </p>
        </div>
        
        <div className="flex gap-2 flex-shrink-0 w-full sm:w-auto">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="glass" size="sm">
                <Download className="h-4 w-4" />
                <span className="ml-2">Export</span>
                <ChevronDown className="h-4 w-4 ml-1" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-48">
              <DropdownMenuItem onClick={handleExportCSV}>
                <Download className="h-4 w-4 mr-2" />
                Export as CSV
              </DropdownMenuItem>
              <DropdownMenuItem onClick={handleExportExcel}>
                <Download className="h-4 w-4 mr-2" />
                Export as Excel
              </DropdownMenuItem>
              <DropdownMenuItem onClick={handleExportPDF}>
                <Download className="h-4 w-4 mr-2" />
                Export as PDF
              </DropdownMenuItem>
              <DropdownMenuItem onClick={handleExportTXT}>
                <Download className="h-4 w-4 mr-2" />
                Export as TXT
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      <div className="flex flex-col sm:flex-row gap-4 items-stretch sm:items-center">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search data..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10"
          />
        </div>
        
        <Select value={sortColumn} onValueChange={setSortColumn}>
          <SelectTrigger className="w-full sm:w-48">
            <SelectValue placeholder="Sort by column" />
          </SelectTrigger>
          <SelectContent>
            {columns.map(column => (
              <SelectItem key={column} value={column}>
                {column}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="rounded-lg border border-border/50 overflow-hidden overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow className="bg-muted/50">
              {columns.map(column => {
                const stats = getColumnStats(column);
                const type = getColumnType(column);
                
                return (
                  <TableHead 
                    key={column}
                    className="cursor-pointer hover:bg-muted/70 transition-colors p-2 sm:p-4 min-w-[120px]"
                    onClick={() => {
                      if (sortColumn === column) {
                        setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
                      } else {
                        setSortColumn(column);
                        setSortDirection('asc');
                      }
                    }}
                  >
                    <div className="space-y-1">
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-medium">{column}</span>
                        <div className="flex gap-1">
                          <Badge variant="secondary" className="text-xs">
                            {type}
                          </Badge>
                          {sortColumn === column && (
                            <Badge variant="default" className="text-xs">
                              {sortDirection === 'asc' ? '↑' : '↓'}
                            </Badge>
                          )}
                        </div>
                      </div>
                      <div className="text-xs text-muted-foreground whitespace-nowrap">
                        {stats.completion}% filled • {stats.unique} unique
                      </div>
                    </div>
                  </TableHead>
                );
              })}
            </TableRow>
          </TableHeader>
          <TableBody>
            {paginatedData.map((row, index) => (
              <TableRow key={index} className="hover:bg-muted/30">
                {columns.map(column => (
                  <TableCell key={column} className="p-2 sm:p-4">
                    <div className="max-w-xs truncate">
                      {row[column] != null ? (
                        typeof row[column] === 'object' 
                          ? JSON.stringify(row[column])
                          : String(row[column])
                      ) : (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </div>
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      {totalPages > 1 && (
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 px-2 sm:px-0">
          <p className="text-xs sm:text-sm text-muted-foreground text-center sm:text-left">
            Showing {((currentPage - 1) * rowsPerPage) + 1} to {Math.min(currentPage * rowsPerPage, sortedData.length)} of {sortedData.length} results
          </p>
          
          <div className="flex items-center gap-1 sm:gap-2 w-full sm:w-auto justify-center">
            <Button 
              variant="outline" 
              size="sm"
              className="px-2 sm:px-4"
              disabled={currentPage === 1}
              onClick={() => setCurrentPage(currentPage - 1)}
            >
              <span className="hidden sm:inline">Previous</span>
              <span className="sm:hidden">Prev</span>
            </Button>
            
            <div className="flex gap-1 overflow-x-auto">
              {Array.from({ length: Math.min(3, totalPages) }, (_, i) => {
                let page;
                if (totalPages <= 3) {
                  page = i + 1;
                } else if (currentPage <= 2) {
                  page = i + 1;
                } else if (currentPage >= totalPages - 1) {
                  page = totalPages - 2 + i;
                } else {
                  page = currentPage - 1 + i;
                }
                
                return (
                  <Button
                    key={page}
                    variant={currentPage === page ? "default" : "outline"}
                    size="sm"
                    className="w-8 h-8 p-0 flex-shrink-0"
                    onClick={() => setCurrentPage(page)}
                  >
                    {page}
                  </Button>
                );
              })}
            </div>
            
            <Button 
              variant="outline" 
              size="sm"
              className="px-2 sm:px-4"
              disabled={currentPage === totalPages}
              onClick={() => setCurrentPage(currentPage + 1)}
            >
              <span className="hidden sm:inline">Next</span>
              <span className="sm:hidden">Next</span>
            </Button>
          </div>
        </div>
      )}
    </Card>
  );
};