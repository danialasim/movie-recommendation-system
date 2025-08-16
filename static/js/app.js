// MovieRec AI - Custom JavaScript

class MovieRecommendationApp {
    constructor() {
        this.selectedMovies = [];
        this.searchTimeout = null;
        this.currentModel = 'sae';
        this.init();
    }

    init() {
        this.bindEvents();
        this.loadModelInfo();
        this.addAnimations();
        this.setupTooltips();
    }

    bindEvents() {
        // Model selection
        $('#model-select').on('change', (e) => {
            this.currentModel = e.target.value;
            this.loadModelInfo();
            this.updateSelectedMoviesDisplay();
        });

        // Movie search
        $('#movie-search').on('input', (e) => this.handleMovieSearch(e.target.value));
        $('#movie-search').on('keydown', (e) => this.handleSearchKeydown(e));

        // Popular movie buttons
        $(document).on('click', '.popular-movie-btn', (e) => {
            e.preventDefault();
            const movieData = {
                id: parseInt($(e.target).data('movie-id')),
                title: $(e.target).data('movie-title')
            };
            this.addMovie(movieData);
        });

        // Remove selected movies
        $(document).on('click', '.movie-tag .btn-close', (e) => {
            e.stopPropagation();
            const movieId = parseInt($(e.target).closest('.movie-tag').data('movie-id'));
            this.removeMovie(movieId);
        });

        // Get recommendations
        $('#get-recommendations').on('click', () => this.getRecommendations());

        // Clear selections
        $('#clear-selections').on('click', () => this.clearSelections());

        // Search result selection
        $(document).on('click', '.search-result-item', (e) => {
            const movieData = {
                id: parseInt($(e.target).data('movie-id')),
                title: $(e.target).data('movie-title')
            };
            this.addMovie(movieData);
            this.clearSearch();
        });

        // Close search results when clicking outside
        $(document).on('click', (e) => {
            if (!$(e.target).closest('.movie-search-container').length) {
                this.clearSearch();
            }
        });

        // Form submission
        $('#recommendation-form').on('submit', (e) => {
            e.preventDefault();
            this.getRecommendations();
        });
    }

    handleMovieSearch(query) {
        clearTimeout(this.searchTimeout);
        
        if (query.length < 2) {
            this.clearSearchResults();
            return;
        }

        this.searchTimeout = setTimeout(() => {
            this.searchMovies(query);
        }, 300);
    }

    handleSearchKeydown(e) {
        const results = $('.search-result-item');
        const current = $('.search-result-item.selected');
        let index = current.length ? results.index(current) : -1;

        switch(e.key) {
            case 'ArrowDown':
                e.preventDefault();
                index = (index + 1) % results.length;
                this.selectSearchResult(index);
                break;
            case 'ArrowUp':
                e.preventDefault();
                index = index <= 0 ? results.length - 1 : index - 1;
                this.selectSearchResult(index);
                break;
            case 'Enter':
                e.preventDefault();
                if (current.length) {
                    current.click();
                }
                break;
            case 'Escape':
                this.clearSearch();
                break;
        }
    }

    selectSearchResult(index) {
        const results = $('.search-result-item');
        results.removeClass('selected');
        if (index >= 0 && index < results.length) {
            $(results[index]).addClass('selected');
        }
    }

    async searchMovies(query) {
        try {
            const response = await fetch(`/movie-search?query=${encodeURIComponent(query)}`);
            const movies = await response.json();
            this.displaySearchResults(movies);
        } catch (error) {
            console.error('Movie search error:', error);
            this.showToast('Error searching movies', 'error');
        }
    }

    displaySearchResults(movies) {
        const container = $('#search-results');
        container.empty();

        if (movies.length === 0) {
            container.html('<div class="text-muted p-3">No movies found</div>');
            return;
        }

        movies.forEach(movie => {
            const isSelected = this.selectedMovies.some(m => m.id === movie.id);
            const resultItem = $(`
                <div class="search-result-item ${isSelected ? 'disabled' : ''}" 
                     data-movie-id="${movie.id}" 
                     data-movie-title="${movie.title}">
                    <span>${movie.title}</span>
                    ${isSelected ? '<small class="text-muted">Already selected</small>' : ''}
                </div>
            `);
            
            if (isSelected) {
                resultItem.css('opacity', '0.5').css('cursor', 'not-allowed');
                resultItem.off('click');
            }
            
            container.append(resultItem);
        });

        container.show();
    }

    clearSearchResults() {
        $('#search-results').empty().hide();
    }

    clearSearch() {
        $('#movie-search').val('');
        this.clearSearchResults();
    }

    addMovie(movieData) {
        // Check if movie is already selected
        if (this.selectedMovies.some(m => m.id === movieData.id)) {
            this.showToast('Movie already selected', 'warning');
            return;
        }

        // Add to selected movies
        this.selectedMovies.push(movieData);
        this.updateSelectedMoviesDisplay();
        this.clearSearch();
        
        // Animation
        this.animateMovieAddition();
    }

    removeMovie(movieId) {
        this.selectedMovies = this.selectedMovies.filter(m => m.id !== movieId);
        this.updateSelectedMoviesDisplay();
    }

    updateSelectedMoviesDisplay() {
        const container = $('#selected-movies-container');
        
        if (this.selectedMovies.length === 0) {
            container.html(`
                <div class="empty-state text-muted text-center">
                    <i class="fas fa-film fa-2x mb-2"></i>
                    <p>No movies selected yet. Search and select movies to get recommendations.</p>
                </div>
            `);
            $('#get-recommendations').prop('disabled', true);
            return;
        }

        const moviesHtml = this.selectedMovies.map(movie => `
            <span class="movie-tag" data-movie-id="${movie.id}">
                ${movie.title}
                <button type="button" class="btn-close ms-2" aria-label="Remove"></button>
            </span>
        `).join('');

        container.html(moviesHtml);
        $('#get-recommendations').prop('disabled', false);
    }

    clearSelections() {
        this.selectedMovies = [];
        this.updateSelectedMoviesDisplay();
        this.clearRecommendations();
        this.animateClearSelections();
    }

    async getRecommendations() {
        if (this.selectedMovies.length === 0) {
            this.showToast('Please select at least one movie', 'warning');
            return;
        }

        this.showLoading('Getting recommendations...');

        try {
            const movieIds = this.selectedMovies.map(m => m.id);
            const response = await fetch('/recommend', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    movie_ids: movieIds,
                    model_type: this.currentModel,
                    num_recommendations: 10
                })
            });

            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }

            this.displayRecommendations(data.recommendations);
            this.scrollToRecommendations();
            
        } catch (error) {
            console.error('Recommendation error:', error);
            this.showToast('Error getting recommendations: ' + error.message, 'error');
        } finally {
            this.hideLoading();
        }
    }

    displayRecommendations(recommendations) {
        const container = $('#recommendations-container');
        
        if (!recommendations || recommendations.length === 0) {
            container.html(`
                <div class="col-12">
                    <div class="alert alert-info">
                        <i class="fas fa-info-circle me-2"></i>
                        No recommendations found. Try selecting different movies.
                    </div>
                </div>
            `);
            return;
        }

        const recommendationsHtml = recommendations.map((movie, index) => {
            const stars = '★'.repeat(Math.floor(movie.rating)) + 
                         '☆'.repeat(5 - Math.floor(movie.rating));
            
            return `
                <div class="col-md-6 col-lg-4 mb-4">
                    <div class="card movie-recommendation-card h-100 fade-in" 
                         style="animation-delay: ${index * 0.1}s">
                        <div class="card-body">
                            <div class="d-flex justify-content-between align-items-start mb-2">
                                <h5 class="card-title">${movie.title}</h5>
                                <span class="rank-badge">#${index + 1}</span>
                            </div>
                            <div class="movie-rating mb-3">
                                <span class="stars">${stars}</span>
                                <span class="score">${movie.rating.toFixed(1)}</span>
                            </div>
                            <div class="d-flex justify-content-between align-items-center">
                                <small class="text-muted">
                                    <i class="fas fa-robot me-1"></i>
                                    Confidence: ${(movie.confidence * 100).toFixed(1)}%
                                </small>
                                <small class="text-muted">
                                    Movie ID: ${movie.movie_id}
                                </small>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        container.html(recommendationsHtml);
        
        // Show recommendations section
        $('#recommendations-section').removeClass('d-none').addClass('slide-up');
    }

    clearRecommendations() {
        $('#recommendations-container').empty();
        $('#recommendations-section').addClass('d-none');
    }

    async loadModelInfo() {
        try {
            const response = await fetch(`/model-info?model_type=${this.currentModel}`);
            const modelInfo = await response.json();
            
            $('#current-model-name').text(modelInfo.name);
            $('#current-model-description').text(modelInfo.description);
            
            // Update model status
            $('#model-status').removeClass('text-success text-warning text-danger')
                              .addClass(modelInfo.status === 'loaded' ? 'text-success' : 'text-warning')
                              .html(modelInfo.status === 'loaded' ? 
                                   '<i class="fas fa-check-circle me-1"></i>Ready' : 
                                   '<i class="fas fa-exclamation-triangle me-1"></i>Loading');
                                   
        } catch (error) {
            console.error('Model info error:', error);
            $('#current-model-name').text('Unknown');
            $('#current-model-description').text('Unable to load model information');
            $('#model-status').removeClass('text-success text-warning')
                              .addClass('text-danger')
                              .html('<i class="fas fa-times-circle me-1"></i>Error');
        }
    }

    showLoading(message = 'Loading...') {
        const overlay = $(`
            <div class="loading-overlay">
                <div class="loading-spinner">
                    <div class="spinner-border text-primary mb-3" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <div>${message}</div>
                </div>
            </div>
        `);
        $('body').append(overlay);
    }

    hideLoading() {
        $('.loading-overlay').remove();
    }

    showToast(message, type = 'info') {
        const alertClass = type === 'error' ? 'alert-danger' : 
                          type === 'warning' ? 'alert-warning' : 
                          type === 'success' ? 'alert-success' : 'alert-info';
        
        const icon = type === 'error' ? 'fas fa-exclamation-circle' : 
                    type === 'warning' ? 'fas fa-exclamation-triangle' : 
                    type === 'success' ? 'fas fa-check-circle' : 'fas fa-info-circle';

        const toast = $(`
            <div class="alert ${alertClass} alert-dismissible fade show position-fixed" 
                 style="top: 100px; right: 20px; z-index: 9999; min-width: 300px;">
                <i class="${icon} me-2"></i>
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `);

        $('body').append(toast);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            toast.alert('close');
        }, 5000);
    }

    scrollToRecommendations() {
        $('html, body').animate({
            scrollTop: $('#recommendations-section').offset().top - 100
        }, 800);
    }

    animateMovieAddition() {
        const lastTag = $('.movie-tag').last();
        lastTag.hide().fadeIn(300);
    }

    animateClearSelections() {
        $('.movie-tag').fadeOut(200, function() {
            $(this).remove();
        });
    }

    addAnimations() {
        // Add entrance animations to existing elements
        $('.hero-section').addClass('fade-in');
        $('.model-status-card').each((index, element) => {
            $(element).css('animation-delay', `${index * 0.1}s`).addClass('slide-up');
        });
        $('.recommendation-card').addClass('slide-up');
    }

    setupTooltips() {
        // Initialize Bootstrap tooltips
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
}

// Initialize app when DOM is ready
$(document).ready(function() {
    window.movieApp = new MovieRecommendationApp();
    
    // Add some popular movies for quick selection
    const popularMovies = [
        { id: 1, title: "Toy Story (1995)" },
        { id: 2, title: "GoldenEye (1995)" },
        { id: 3, title: "Four Weddings and a Funeral (1994)" },
        { id: 4, title: "Get Shorty (1995)" },
        { id: 5, title: "Copycat (1995)" },
        { id: 6, title: "Shanghai Triad (Yao a yao yao dao waipo qiao) (1995)" },
        { id: 7, title: "Twelve Monkeys (1995)" },
        { id: 8, title: "Babe (1995)" },
        { id: 9, title: "Dead Man Walking (1995)" },
        { id: 10, title: "Richard III (1995)" }
    ];

    // Add popular movies to the page
    const popularMoviesHtml = popularMovies.map(movie => `
        <button type="button" class="btn btn-outline-primary btn-sm popular-movie-btn me-2 mb-2" 
                data-movie-id="${movie.id}" data-movie-title="${movie.title}">
            ${movie.title}
        </button>
    `).join('');
    
    $('#popular-movies').html(popularMoviesHtml);
});

// Smooth scrolling for anchor links
$(document).on('click', 'a[href^="#"]', function (event) {
    event.preventDefault();
    
    $('html, body').animate({
        scrollTop: $($.attr(this, 'href')).offset().top - 80
    }, 500);
});

// Add loading state to forms
$(document).on('submit', 'form', function() {
    const submitBtn = $(this).find('button[type="submit"]');
    submitBtn.prop('disabled', true);
    
    setTimeout(() => {
        submitBtn.prop('disabled', false);
    }, 2000);
});

// Add hover effects to cards
$(document).on('mouseenter', '.card', function() {
    $(this).addClass('shadow-lg');
}).on('mouseleave', '.card', function() {
    $(this).removeClass('shadow-lg');
});

// Keyboard shortcuts
$(document).keydown(function(e) {
    // Ctrl/Cmd + K to focus search
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        $('#movie-search').focus();
    }
    
    // Escape to clear search
    if (e.key === 'Escape') {
        $('#movie-search').blur();
        window.movieApp.clearSearch();
    }
});

// Add intersection observer for animations
if ('IntersectionObserver' in window) {
    const animateOnScroll = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate');
            }
        });
    });

    $(document).on('DOMNodeInserted', '.slide-up, .fade-in', function() {
        animateOnScroll.observe(this);
    });
}
