'use client';

import { useState, useEffect } from 'react';
import Image from 'next/image';

interface MovieData {
  title: string;
  overview: string;
  original_tagline: string;
  genres: string[];
  poster_url: string | null;
  generated_taglines: {
    baseline: string;
    rag_infer: string;
    genre_only: string;
    genre_rag: string;
    genre_rag_boosted: string;
    genre_boosted: string;
  };
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

export default function Home() {
  const [movies, setMovies] = useState<string[]>([]);
  const [selectedMovie, setSelectedMovie] = useState<string>('');
  const [movieData, setMovieData] = useState<MovieData | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingMovies, setLoadingMovies] = useState(true);

  useEffect(() => {
    fetchMovies();
  }, []);

  const fetchMovies = async () => {
    try {
      setLoadingMovies(true);
      const response = await fetch(`${API_URL}/movies`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setMovies(data);
    } catch (error) {
      console.error('Error fetching movies:', error);
      // Show user-friendly error message
      setMovies([]);
    } finally {
      setLoadingMovies(false);
    }
  };

  const fetchMovieData = async (title: string) => {
    if (!title) return;
    
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/movie/${encodeURIComponent(title)}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      
      // Validate the data structure
      if (data && typeof data === 'object') {
        setMovieData(data);
      } else {
        console.error('Invalid movie data received:', data);
        setMovieData(null);
      }
    } catch (error) {
      console.error('Error fetching movie data:', error);
      setMovieData(null);
    } finally {
      setLoading(false);
    }
  };

  const handleMovieSelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const title = e.target.value;
    setSelectedMovie(title);
    fetchMovieData(title);
  };

  const TaglineCard = ({ title, content, bgColor, borderColor }: {
    title: string;
    content: string;
    bgColor: string;
    borderColor: string;
  }) => (
    <div className="mb-6">
      <h3 className="text-lg font-semibold mb-3 text-gray-200">{title}</h3>
      <div
        className={`${bgColor} p-4 border-l-4 ${borderColor} rounded-lg`}
        style={{ fontSize: '16px' }}
      >
        <span className="text-gray-100">{content}</span>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-900 pb-16">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">🍿 Tagline Generator</h1>
          <p className="text-gray-300">Generate movie taglines with fine-tuned GPT-2, RAG and genre conditioning</p>
        </div>

        <div className="max-w-4xl mx-auto">
          {loadingMovies ? (
            <div className="text-center py-12">
              <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mb-4"></div>
              <p className="text-lg text-gray-300">Loading movie database...</p>
              <p className="text-sm text-gray-400 mt-2">This may take a moment on first load</p>
            </div>
          ) : movies.length === 0 ? (
            <div className="text-center py-12">
              <div className="text-red-400 text-lg mb-2">Unable to load movies</div>
              <p className="text-gray-400">Please check if the backend API is running</p>
              <button 
                onClick={fetchMovies}
                className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Retry
              </button>
            </div>
          ) : (
            <div className="bg-gray-800 rounded-lg shadow-lg p-6 mb-8 border border-gray-700">
              <label htmlFor="movie-select" className="block text-sm font-medium text-gray-300 mb-2">
                Choose a movie:
              </label>
              <select
                id="movie-select"
                value={selectedMovie}
                onChange={handleMovieSelect}
                className="w-full p-3 border border-gray-600 bg-gray-700 text-white rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="">Select a movie...</option>
                {movies.map((movie) => (
                  <option key={movie} value={movie}>
                    {movie}
                  </option>
                ))}
              </select>
            </div>
          )}

          {loading && (
            <div className="text-center py-12">
              <div className="inline-block animate-spin rounded-full h-10 w-10 border-b-2 border-blue-500 mb-4"></div>
              <p className="text-lg text-gray-300">Loading movie details...</p>
            </div>
          )}

          {movieData && !loading && (
            <div className="bg-gray-800 rounded-lg shadow-lg p-6 border border-gray-700">
              <div className="grid md:grid-cols-3 gap-6 mb-8">
                <div className="md:col-span-1">
                  {movieData.poster_url ? (
                    <div className="text-center">
                      <Image
                        src={movieData.poster_url}
                        alt={`${movieData.title} poster`}
                        width={300}
                        height={450}
                        className="rounded-lg shadow-md mx-auto"
                      />
                      <p className="text-sm text-gray-400 mt-2">Poster</p>
                    </div>
                  ) : (
                    <div className="bg-gray-700 w-full h-64 rounded-lg flex items-center justify-center">
                      <p className="text-gray-400">No poster available</p>
                    </div>
                  )}
                </div>

                <div className="md:col-span-2">
                  <h2 className="text-3xl font-bold text-white mb-4">{movieData.title}</h2>
                  
                  <div className="mb-6">
                    <h3 className="text-lg font-semibold mb-2 text-gray-200">Genres</h3>
                    <div className="flex flex-wrap gap-2">
                      {(movieData.genres || []).map((genre, index) => (
                        <span
                          key={index}
                          className="bg-red-600 text-white px-3 py-1 rounded-full text-sm"
                        >
                          {genre}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div className="mb-6">
                    <h3 className="text-lg font-semibold mb-2 text-gray-200">Overview</h3>
                    <p className="text-gray-300 leading-relaxed">{movieData.overview}</p>
                  </div>

                  <div className="mb-6">
                    <h3 className="text-lg font-semibold mb-2 text-gray-200">Original Tagline</h3>
                    <p className="text-gray-300 font-medium">{movieData.original_tagline}</p>
                  </div>
                </div>
              </div>

              <div className="border-t border-gray-700 pt-6">
                <h2 className="text-2xl font-bold text-white mb-6">Generated Taglines</h2>
                
                <TaglineCard
                  title="Baseline Model Generated Tagline"
                  content={movieData.generated_taglines?.baseline || 'No tagline available'}
                  bgColor="bg-yellow-900"
                  borderColor="border-yellow-400"
                />

                <TaglineCard
                  title="RAG At Inference Generated Tagline"
                  content={movieData.generated_taglines?.rag_infer || 'No tagline available'}
                  bgColor="bg-green-900"
                  borderColor="border-green-400"
                />

                <TaglineCard
                  title="Overview + Genre Model Generated Tagline"
                  content={movieData.generated_taglines?.genre_only || 'No tagline available'}
                  bgColor="bg-blue-900"
                  borderColor="border-blue-400"
                />

                <TaglineCard
                  title="Overview + Genre + RAG Model Generated Tagline"
                  content={movieData.generated_taglines?.genre_rag || 'No tagline available'}
                  bgColor="bg-purple-900"
                  borderColor="border-purple-400"
                />

                <TaglineCard
                  title="Overview + Genre + RAG (Boosted) Model Generated Tagline"
                  content={movieData.generated_taglines?.genre_rag_boosted || 'No tagline available'}
                  bgColor="bg-orange-900"
                  borderColor="border-orange-400"
                />

                <TaglineCard
                  title="Overview + Genre (Boosted) Model Generated Tagline"
                  content={movieData.generated_taglines?.genre_boosted || 'No tagline available'}
                  bgColor="bg-teal-900"
                  borderColor="border-teal-400"
                />
              </div>
            </div>
          )}
        </div>
      </div>
      
      {/* Credit Footer */}
      <footer className="fixed bottom-0 left-0 right-0 bg-gray-800 border-t border-gray-700 py-3">
        <div className="container mx-auto px-4 text-center">
          <p className="text-sm text-gray-400">
            Developed by <span className="text-white font-medium">Cam Nguyen</span> and <span className="text-white font-medium">Ben Wyman</span> © 2025
          </p>
        </div>
      </footer>
    </div>
  );
}
