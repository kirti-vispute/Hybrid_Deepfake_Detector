const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api';

async function parseErrorResponse(response) {
  try {
    const data = await response.json();
    if (data?.message) {
      if (data.details) {
        return `${data.message} ${data.details}`;
      }
      return data.message;
    }
  } catch (error) {
    // no-op
  }
  return `The server returned an error (HTTP ${response.status}).`;
}

export async function fetchModelAvailability() {
  const response = await fetch(`${API_BASE_URL}/models`);
  if (!response.ok) {
    throw new Error(await parseErrorResponse(response));
  }
  const payload = await response.json();
  const available = payload?.available || {};

  return {
    ...payload,
    available: {
      ...available,
      // Treat classical availability as a valid hybrid-serving fallback.
      hybrid: Boolean(available.hybrid || available.classical),
    },
  };
}

export async function predictImage({ file }) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('model', 'hybrid');

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await parseErrorResponse(response));
  }

  return response.json();
}

export { API_BASE_URL };
