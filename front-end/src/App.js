import React, { useState, useEffect, useCallback } from 'react';

// --- Configuration ---
const PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Never underestimate the power of a good book.",
    "The early bird catches the worm.",
    "To be or not to be, that is the question.",
    "I think, therefore I am.",
    "The journey of a thousand miles begins with a single step."
];

// --- Sub-components ---

const UserIdStep = ({ onStart }) => {
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

    return (
        <div>
            <label htmlFor="userId" className="block text-sm font-medium text-gray-700 mb-2">Please enter your name or a unique ID:</label>
            <input
                type="text"
                id="userId"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                onKeyPress={handleKeyPress}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                placeholder="e.g., noah_01"
            />
            <button
                onClick={handleStart}
                disabled={!userId.trim()}
                className="mt-4 w-full bg-indigo-600 text-white font-semibold py-2 px-4 rounded-lg hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-gray-400 disabled:cursor-not-allowed"
            >
                Start Session
            </button>
        </div>
    );
};

const CollectionStep = ({ currentUser, onComplete }) => {
    const [currentPromptIndex, setCurrentPromptIndex] = useState(0);
    const [typingInput, setTypingInput] = useState('');
    const [keystrokeData, setKeystrokeData] = useState([]);
    const [keyPressTimes, setKeyPressTimes] = useState({});
    const [lastEvent, setLastEvent] = useState(null); // NEW: To track the previous key event
    const [statusMessage, setStatusMessage] = useState('');

    const displayNextPrompt = useCallback(() => {
        if (currentPromptIndex >= PROMPTS.length - 1) {
             onComplete(keystrokeData);
        } else {
            setCurrentPromptIndex(prevIndex => prevIndex + 1);
            setTypingInput('');
            setStatusMessage('');
            setLastEvent(null); // NEW: Reset for the new prompt
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
        
        // --- NEW: Calculate all timing metrics ---
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
        setLastEvent(newEvent); // NEW: Update the last event
        
        // Clean up the press time map
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
                className="w-full text-lg px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500"
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
    const [step, setStep] = useState('userId');
    const [currentUser, setCurrentUser] = useState('');
    const [dataSent, setDataSent] = useState(false);

    const handleStartSession = (userId) => {
        setCurrentUser(userId);
        setStep('collection');
    };

    const handleSessionComplete = async (finalData) => {
        setStep('pending');
        
        if (finalData.length === 0) {
            setStep('complete');
            return;
        }

        // --- NEW: Updated headers for the CSV file ---
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
    
    const renderStep = () => {
        switch (step) {
            case 'userId':
                return <UserIdStep onStart={handleStartSession} />;
            case 'collection':
                return <CollectionStep currentUser={currentUser} onComplete={handleSessionComplete} />;
            case 'complete':
                 return <CompleteStep dataSent={dataSent} />;
            case 'pending':
                return <p className="text-center">Sending data to server...</p>;
            default:
                return <UserIdStep onStart={handleStartSession} />;
        }
    };

    return (
        <div className="bg-gray-100 flex items-center justify-center min-h-screen p-4 font-sans">
            <div className="w-full max-w-2xl mx-auto">
                <div className="bg-white rounded-2xl shadow-lg p-6 md:p-8">
                    <header className="text-center mb-6">
                        <h1 className="text-2xl md:text-3xl font-bold text-gray-800">Keystroke Dynamics Collector</h1>
                        <p className="text-gray-500 mt-1">Help us understand typing patterns.</p>
                    </header>
                    <main>
                        {renderStep()}
                    </main>
                </div>
                <footer className="text-center mt-4">
                    <p className="text-xs text-gray-500">All data is sent securely to the server.</p>
                </footer>
            </div>
        </div>
    );
}
