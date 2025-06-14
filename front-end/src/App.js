import React, { useState, useCallback } from 'react';

// --- Configuration ---
const PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Never underestimate the power of a good book.",
    "The early bird catches the worm.",
    "To be or not to be, that is the question.",
    "I think, therefore I am.",
    "The journey of a thousand miles begins with a single step."
];

const AUTH_PROMPTS = [
    "Please type this sentence to verify your identity.",
    "Authentication requires your unique typing pattern.",
    "Type this phrase to confirm it's really you."
];

// --- Sub-components ---

const ModeSelection = ({ onModeSelect }) => {
    return (
        <div className="text-center">
            <h2 className="text-xl font-bold text-gray-800 mb-6">Choose Your Action</h2>
            <div className="space-y-4">
                <button
                    onClick={() => onModeSelect('collect')}
                    className="w-full bg-indigo-600 text-white font-semibold py-3 px-6 rounded-lg hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors"
                >
                    📝 Collect Typing Data
                    <p className="text-sm text-indigo-200 mt-1">Record your typing patterns for training</p>
                </button>
                <button
                    onClick={() => onModeSelect('authenticate')}
                    className="w-full bg-green-600 text-white font-semibold py-3 px-6 rounded-lg hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 transition-colors"
                >
                    🔐 Sign In & Authenticate
                    <p className="text-sm text-green-200 mt-1">Test your identity using keystroke patterns</p>
                </button>
            </div>
        </div>
    );
};

const UserIdStep = ({ onStart, mode }) => {
    const [userId, setUserId] = useState('');

    const handleStart = () => {
        if (userId.trim()) {
            onStart(userId.trim());
        }
    };
    
    const handleKeyPress = (e) => {
        if (e.key === 'Enter') {
            handleStart();
        }
    };

    const isAuthMode = mode === 'authenticate';

    return (
        <div>
            <label htmlFor="userId" className="block text-sm font-medium text-gray-700 mb-2">
                {isAuthMode ? 'Enter your username to sign in:' : 'Please enter your name or a unique ID:'}
            </label>
            <input
                type="text"
                id="userId"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                onKeyPress={handleKeyPress}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                placeholder={isAuthMode ? "e.g., noah" : "e.g., noah_01"}
            />
            <button
                onClick={handleStart}
                disabled={!userId.trim()}
                className={`mt-4 w-full font-semibold py-2 px-4 rounded-lg focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:bg-gray-400 disabled:cursor-not-allowed ${
                    isAuthMode 
                        ? 'bg-green-600 text-white hover:bg-green-700 focus:ring-green-500' 
                        : 'bg-indigo-600 text-white hover:bg-indigo-700 focus:ring-indigo-500'
                }`}
            >
                {isAuthMode ? 'Sign In' : 'Start Session'}
            </button>
        </div>
    );
};

const AuthenticationStep = ({ currentUser, onComplete }) => {
    const [currentPromptIndex, setCurrentPromptIndex] = useState(0);
    const [typingInput, setTypingInput] = useState('');
    const [keystrokeData, setKeystrokeData] = useState([]);
    const [keyPressTimes, setKeyPressTimes] = useState({});
    const [lastEvent, setLastEvent] = useState(null);
    const [statusMessage, setStatusMessage] = useState('');
    const [isAuthenticating, setIsAuthenticating] = useState(false);

    const handleAuthentication = useCallback(async () => {
        if (keystrokeData.length === 0) {
            setStatusMessage('No keystroke data collected. Please type the phrase first.');
            return;
        }

        setIsAuthenticating(true);
        setStatusMessage('Authenticating...');

        try {
            const response = await fetch('http://127.0.0.1:5000/api/authenticate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    username: currentUser,
                    keystroke_data: keystrokeData,
                    threshold: 0.5
                }),
            });

            const result = await response.json();
            
            if (response.ok) {
                onComplete({
                    success: true,
                    authenticated: result.authenticated,
                    confidence: result.confidence,
                    message: result.message,
                    details: result
                });
            } else {
                onComplete({
                    success: false,
                    error: result.error,
                    authenticated: false,
                    confidence: 0
                });
            }
        } catch (error) {
            console.error('Authentication error:', error);
            onComplete({
                success: false,
                error: 'Failed to connect to authentication server',
                authenticated: false,
                confidence: 0
            });
        } finally {
            setIsAuthenticating(false);
        }
    }, [keystrokeData, currentUser, onComplete]);

    const handleKeyDown = (e) => {
        if (e.key.length > 1 && e.key !== 'Backspace' && e.key !== ' ') return;
        const key = e.key;
        if (key in keyPressTimes) return;
        setKeyPressTimes(prev => ({ ...prev, [key]: performance.now() }));
    };

    const handleKeyUp = useCallback((e) => {
        const key = e.key;

        if (key === 'Enter') {
            if (typingInput.trim().length > 5) {
                handleAuthentication();
            } else {
                setStatusMessage('Please type the full phrase before pressing Enter.');
            }
            return;
        }

        if (e.key.length > 1 && e.key !== 'Backspace' && e.key !== ' ') return;
        
        const pressTime = keyPressTimes[key];
        if (!pressTime) return;

        const releaseTime = performance.now();
        
        const dwell_time = releaseTime - pressTime;
        let p2p_time = 0;
        let r2p_time = 0;
        let r2r_time = 0;

        if (lastEvent) {
            p2p_time = pressTime - lastEvent.press_time;
            r2p_time = pressTime - lastEvent.release_time;
            r2r_time = releaseTime - lastEvent.release_time;
        }

        const newEvent = {
            user: currentUser,
            prompt: AUTH_PROMPTS[currentPromptIndex],
            key: key,
            press_time: pressTime,
            release_time: releaseTime,
            dwell_time: dwell_time,
            p2p_time: p2p_time,
            r2p_time: r2p_time,
            r2r_time: r2r_time,
        };

        setKeystrokeData(prevData => [...prevData, newEvent]);
        setLastEvent(newEvent);
        
        setKeyPressTimes(prev => {
            const newTimes = { ...prev };
            delete newTimes[key];
            return newTimes;
        });

    }, [typingInput, keyPressTimes, lastEvent, currentUser, currentPromptIndex, handleAuthentication]);

    return (
        <div>
            <div className="text-center mb-6">
                <div className="inline-flex items-center px-4 py-2 bg-green-100 text-green-800 rounded-full text-sm font-medium mb-4">
                    🔐 Authentication Mode
                </div>
                <p className="text-sm text-gray-600">Signing in as: <strong>{currentUser}</strong></p>
            </div>
            
            <p className="text-lg md:text-xl text-center font-mono bg-gray-100 p-4 rounded-lg mb-4">
                {AUTH_PROMPTS[currentPromptIndex]}
            </p>
            
            <input
                type="text"
                value={typingInput}
                onChange={(e) => setTypingInput(e.target.value)}
                onKeyDown={handleKeyDown}
                onKeyUp={handleKeyUp}
                className="w-full text-lg px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500"
                placeholder="Type the phrase above and press Enter to authenticate..."
                autoComplete="off"
                autoCorrect="off"
                autoCapitalize="off"
                spellCheck="false"
                autoFocus
                disabled={isAuthenticating}
            />
            
            <div className="mt-4 text-center">
                <button
                    onClick={handleAuthentication}
                    disabled={isAuthenticating || keystrokeData.length === 0}
                    className="bg-green-600 text-white font-semibold py-2 px-6 rounded-lg hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:bg-gray-400 disabled:cursor-not-allowed"
                >
                    {isAuthenticating ? 'Authenticating...' : 'Authenticate Now'}
                </button>
            </div>
            
            <p className="text-center text-sm text-red-600 mt-2 font-semibold h-5">{statusMessage}</p>
            
            {keystrokeData.length > 0 && (
                <div className="mt-4 text-center text-sm text-gray-600">
                    Captured {keystrokeData.length} keystrokes
                </div>
            )}
        </div>
    );
};

const CollectionStep = ({ currentUser, onComplete }) => {
    const [currentPromptIndex, setCurrentPromptIndex] = useState(0);
    const [typingInput, setTypingInput] = useState('');
    const [keystrokeData, setKeystrokeData] = useState([]);
    const [keyPressTimes, setKeyPressTimes] = useState({});
    const [lastEvent, setLastEvent] = useState(null);
    const [statusMessage, setStatusMessage] = useState('');

    const displayNextPrompt = useCallback(() => {
        if (currentPromptIndex >= PROMPTS.length - 1) {
             onComplete(keystrokeData);
        } else {
            setCurrentPromptIndex(prevIndex => prevIndex + 1);
            setTypingInput('');
            setStatusMessage('');
            setLastEvent(null);
        }
    }, [currentPromptIndex, onComplete, keystrokeData]);
    
    const handleKeyDown = (e) => {
        if (e.key.length > 1 && e.key !== 'Backspace' && e.key !== ' ') return;
        const key = e.key;
        if (key in keyPressTimes) return;
        setKeyPressTimes(prev => ({ ...prev, [key]: performance.now() }));
    };

    const handleKeyUp = useCallback((e) => {
        const key = e.key;

        if (key === 'Enter') {
            if (typingInput.trim().length > 5) {
                displayNextPrompt();
            } else {
                setStatusMessage('Please type the full phrase before pressing Enter.');
            }
            return;
        }

        if (e.key.length > 1 && e.key !== 'Backspace' && e.key !== ' ') return;
        
        const pressTime = keyPressTimes[key];
        if (!pressTime) return;

        const releaseTime = performance.now();
        
        const dwell_time = releaseTime - pressTime;
        let p2p_time = 0;
        let r2p_time = 0;
        let r2r_time = 0;

        if (lastEvent) {
            p2p_time = pressTime - lastEvent.press_time;
            r2p_time = pressTime - lastEvent.release_time;
            r2r_time = releaseTime - lastEvent.release_time;
        }

        const newEvent = {
            user: currentUser,
            prompt: PROMPTS[currentPromptIndex],
            key: key,
            press_time: pressTime,
            release_time: releaseTime,
            dwell_time: dwell_time,
            p2p_time: p2p_time,
            r2p_time: r2p_time,
            r2r_time: r2r_time,
        };

        setKeystrokeData(prevData => [...prevData, newEvent]);
        setLastEvent(newEvent);
        
        setKeyPressTimes(prev => {
            const newTimes = { ...prev };
            delete newTimes[key];
            return newTimes;
        });

    }, [typingInput, keyPressTimes, lastEvent, currentUser, currentPromptIndex, displayNextPrompt]);

    const progress = ((currentPromptIndex + 1) / PROMPTS.length) * 100;

    return (
        <div>
            <div className="text-center mb-4">
                <div className="inline-flex items-center px-4 py-2 bg-indigo-100 text-indigo-800 rounded-full text-sm font-medium mb-4">
                    📝 Data Collection Mode
                </div>
                <p className="text-sm font-medium text-gray-600">Prompt {currentPromptIndex + 1} of {PROMPTS.length}</p>
                <div className="mt-2 bg-gray-200 rounded-full h-2.5 w-full">
                    <div className="bg-indigo-600 h-2.5 rounded-full" style={{ width: `${progress}%` }}></div>
                </div>
            </div>
            <p className="text-lg md:text-xl text-center font-mono bg-gray-100 p-4 rounded-lg mb-4">{PROMPTS[currentPromptIndex]}</p>
            <input
                type="text"
                value={typingInput}
                onChange={(e) => setTypingInput(e.target.value)}
                onKeyDown={handleKeyDown}
                onKeyUp={handleKeyUp}
                className="w-full text-lg px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                placeholder="Start typing the phrase here..."
                autoComplete="off"
                autoCorrect="off"
                autoCapitalize="off"
                spellCheck="false"
                autoFocus
            />
            <p className="text-center text-sm text-red-600 mt-2 font-semibold h-5">{statusMessage}</p>
        </div>
    );
};

const AuthenticationResult = ({ result, onRestart }) => {
    const { success, authenticated, confidence, message, error } = result;
    
    return (
        <div className="text-center">
            {success ? (
                <div>
                    {authenticated ? (
                        <div>
                            <svg className="mx-auto h-16 w-16 text-green-500 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            <h2 className="text-2xl font-bold text-green-800 mb-2">✅ Authentication Successful!</h2>
                            <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-4">
                                <p className="text-lg font-semibold text-green-800">
                                    Confidence: {confidence.toFixed(1)}%
                                </p>
                                <p className="text-sm text-green-600 mt-1">{message}</p>
                            </div>
                        </div>
                    ) : (
                        <div>
                            <svg className="mx-auto h-16 w-16 text-red-500 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
                            </svg>
                            <h2 className="text-2xl font-bold text-red-800 mb-2">❌ Authentication Failed</h2>
                            <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
                                <p className="text-lg font-semibold text-red-800">
                                    Confidence: {confidence.toFixed(1)}%
                                </p>
                                <p className="text-sm text-red-600 mt-1">{message}</p>
                            </div>
                        </div>
                    )}
                </div>
            ) : (
                <div>
                    <svg className="mx-auto h-16 w-16 text-yellow-500 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
                    </svg>
                    <h2 className="text-2xl font-bold text-yellow-800 mb-2">⚠️ Authentication Error</h2>
                    <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-4">
                        <p className="text-sm text-yellow-800">{error}</p>
                    </div>
                </div>
            )}
            
            <button
                onClick={onRestart}
                className="mt-4 bg-gray-600 text-white font-semibold py-2 px-6 rounded-lg hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500"
            >
                Try Again
            </button>
        </div>
    );
};

const CompleteStep = ({ dataSent }) => (
    <div className="text-center">
        <svg className="mx-auto h-12 w-12 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <h2 className="mt-4 text-2xl font-bold text-gray-800">Session Complete!</h2>
        <p className="text-gray-600 mt-1">Thank you for your participation.</p>
        {dataSent && <p className="text-sm text-green-600 mt-2 font-semibold">Data successfully sent to server.</p>}
    </div>
);

// --- Main App Component ---
export default function App() {
    const [mode, setMode] = useState(null); // 'collect' or 'authenticate'
    const [step, setStep] = useState('mode');
    const [currentUser, setCurrentUser] = useState('');
    const [dataSent, setDataSent] = useState(false);
    const [authResult, setAuthResult] = useState(null);

    const handleModeSelect = (selectedMode) => {
        setMode(selectedMode);
        setStep('userId');
    };

    const handleStartSession = (userId) => {
        setCurrentUser(userId);
        if (mode === 'collect') {
            setStep('collection');
        } else if (mode === 'authenticate') {
            setStep('authentication');
        }
    };

    const handleAuthenticationComplete = (result) => {
        setAuthResult(result);
        setStep('authResult');
    };

    const handleSessionComplete = async (finalData) => {
        setStep('pending');
        
        if (finalData.length === 0) {
            setStep('complete');
            return;
        }

        const headers = ['user', 'prompt', 'key', 'press_time', 'release_time', 'dwell_time', 'p2p_time', 'r2p_time', 'r2r_time'];
        const csvRows = finalData.map(row => {
            const sanitizedPrompt = `"${row.prompt.replace(/"/g, '""')}"`;
            const key = row.key === ',' ? '","' : row.key;
            return [row.user, sanitizedPrompt, key, row.press_time, row.release_time, row.dwell_time, row.p2p_time, row.r2p_time, row.r2r_time].join(',');
        });
        const csvContent = [headers.join(','), ...csvRows].join('\n');
        
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const formData = new FormData();
        formData.append('file', blob, `${currentUser}_keystrokes.csv`);

        try {
            const response = await fetch('http://127.0.0.1:5000/api/save_data', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            
            setDataSent(true);
        } catch (error) {
            console.error("Failed to send data to backend:", error);
            setDataSent(false);
        } finally {
            setStep('complete');
        }
    };

    const handleRestart = () => {
        setMode(null);
        setStep('mode');
        setCurrentUser('');
        setDataSent(false);
        setAuthResult(null);
    };
    
    const renderStep = () => {
        switch (step) {
            case 'mode':
                return <ModeSelection onModeSelect={handleModeSelect} />;
            case 'userId':
                return <UserIdStep onStart={handleStartSession} mode={mode} />;
            case 'collection':
                return <CollectionStep currentUser={currentUser} onComplete={handleSessionComplete} />;
            case 'authentication':
                return <AuthenticationStep currentUser={currentUser} onComplete={handleAuthenticationComplete} />;
            case 'authResult':
                return <AuthenticationResult result={authResult} onRestart={handleRestart} />;
            case 'complete':
                return <CompleteStep dataSent={dataSent} />;
            case 'pending':
                return <p className="text-center">Sending data to server...</p>;
            default:
                return <ModeSelection onModeSelect={handleModeSelect} />;
        }
    };

    const getTitle = () => {
        if (mode === 'authenticate') {
            return 'Keystroke Authentication System';
        } else if (mode === 'collect') {
            return 'Keystroke Dynamics Collector';
        }
        return 'Keystroke Dynamics System';
    };

    const getSubtitle = () => {
        if (mode === 'authenticate') {
            return 'Secure sign-in using your typing patterns';
        } else if (mode === 'collect') {
            return 'Help us understand typing patterns';
        }
        return 'Advanced biometric authentication';
    };

    return (
        <div className="bg-gray-100 flex items-center justify-center min-h-screen p-4 font-sans">
            <div className="w-full max-w-2xl mx-auto">
                <div className="bg-white rounded-2xl shadow-lg p-6 md:p-8">
                    <header className="text-center mb-6">
                        <h1 className="text-2xl md:text-3xl font-bold text-gray-800">{getTitle()}</h1>
                        <p className="text-gray-500 mt-1">{getSubtitle()}</p>
                        {step !== 'mode' && (
                            <button
                                onClick={handleRestart}
                                className="mt-2 text-sm text-gray-500 hover:text-gray-700 underline"
                            >
                                ← Back to main menu
                            </button>
                        )}
                    </header>
                    <main>
                        {renderStep()}
                    </main>
                </div>
                <footer className="text-center mt-4">
                    <p className="text-xs text-gray-500">
                        {mode === 'authenticate' 
                            ? 'Your typing patterns are processed securely.' 
                            : 'All data is sent securely to the server.'}
                    </p>
                </footer>
            </div>
        </div>
    );
}
