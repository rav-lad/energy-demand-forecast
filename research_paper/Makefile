# Makefile for compiling LaTeX research paper

# Main document
MAIN = energy_trading_research
TEX = $(MAIN).tex
PDF = $(MAIN).pdf

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Compilation flags
LATEXFLAGS = -interaction=nonstopmode -halt-on-error

# Default target
all: $(PDF)

# Compile PDF (run twice for references)
$(PDF): $(TEX)
	@echo "Compiling LaTeX document..."
	$(LATEX) $(LATEXFLAGS) $(TEX)
	@echo "Second pass for references..."
	$(LATEX) $(LATEXFLAGS) $(TEX)
	@echo "PDF generated: $(PDF)"

# Clean auxiliary files
clean:
	@echo "Cleaning auxiliary files..."
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.lot *.lof
	@echo "Cleaned."

# Clean everything including PDF
distclean: clean
	@echo "Removing PDF..."
	rm -f $(PDF)
	@echo "All cleaned."

# View PDF (Linux)
view: $(PDF)
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(PDF); \
	elif command -v open > /dev/null; then \
		open $(PDF); \
	else \
		echo "No PDF viewer found. Please open $(PDF) manually."; \
	fi

# Help
help:
	@echo "Makefile for LaTeX Research Paper"
	@echo ""
	@echo "Usage:"
	@echo "  make          - Compile the PDF"
	@echo "  make clean    - Remove auxiliary files"
	@echo "  make distclean - Remove all generated files including PDF"
	@echo "  make view     - Open the PDF (Linux/macOS)"
	@echo "  make help     - Show this help message"

.PHONY: all clean distclean view help
