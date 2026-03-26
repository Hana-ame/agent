async function fetchUserData(userId) {
    try {
        const response = await fetch(`https://api.example.com/users/${userId}`);
        if (!response.ok) throw new Error('Network error');
        const data = await response.json();
        console.log('User:', data);
        return data;
    } catch (error) {
        console.error('Fetch failed:', error);
    }
}