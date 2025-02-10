import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Bluetooth, Radio, Wifi } from 'lucide-react';

const DeviceList = ({ devices }) => {
  return (
    <div className="w-full">
      <h2 className="text-xl font-bold mb-4">Detected Devices</h2>
      <div className="space-y-4">
        {devices.map(device => (
          <div key={device.id} className="space-y-2">
            <div>
              <h3 className="text-lg font-medium">{device.name}</h3>
              <p className="text-sm text-gray-300">Seat: {device.predicted_seat}</p>
              <p className="text-sm text-gray-300">ID: {device.id}</p>
              <p className="text-sm text-gray-300">{device.device_type}</p>
            </div>
            <div className="grid grid-cols-2 gap-4">
              {Object.entries(device.rssi_values).map(([sensor, value]) => (
                <div key={sensor} className="flex justify-between">
                  <span className="text-gray-400">{sensor}:</span>
                  <span className="text-gray-200">{value.toFixed(1)} dBm</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

const ArenaGrid = ({ devices, sectionColors }) => {
  const rows = 8;
  const cols = 11;
  const devicesBySeat = {};
  
  devices.forEach(device => {
    if (device.predicted_seat) {
      if (!devicesBySeat[device.predicted_seat]) {
        devicesBySeat[device.predicted_seat] = [];
      }
      devicesBySeat[device.predicted_seat].push(device);
    }
  });

  const seats = [];
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const seatId = `${String.fromCharCode(65 + row)}${col + 1}`;
      seats.push({
        id: seatId,
        section: seatId[0],
        devices: devicesBySeat[seatId] || []
      });
    }
  }

  return (
    <div className="w-full">
      <h2 className="text-xl font-bold mb-4">Arena Layout</h2>
      <div className="grid grid-cols-2 gap-x-4">
        {seats.map((seat) => (
          <div
            key={seat.id}
            className="h-8 mb-2 rounded flex items-center px-3 text-sm font-medium"
            style={{
              backgroundColor: sectionColors[seat.section],
              opacity: seat.devices.length ? 1 : 0.7,
              width: 30,
              height: 30,
              color: 'black',
              gridColumn: (Number(seat.id.slice(1))-1) % 11 + 1

            }}
          >
            {seat.id}
            {seat.devices.length > 0 && (
              <span className="ml-2">({seat.devices.length})</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

const SignalStrengthChart = ({ devices }) => {
  const chartData = devices.map(device => ({
    name: device.name || device.id,
    ...device.rssi_values
  }));

  return (
    <div className="w-full">
      <h2 className="text-xl font-bold mb-4">Signal Strength Analysis</h2>
      <div className="h-[300px] bg-gray-800/30 rounded p-4">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid stroke="#2D3748" strokeDasharray="3 3" />
            <XAxis 
              dataKey="name" 
              stroke="#A0AEC0" 
              tick={{ fill: '#A0AEC0' }}
            />
            <YAxis 
              domain={[-100, -20]} 
              stroke="#A0AEC0"
              tick={{ fill: '#A0AEC0' }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1A202C',
                border: 'none',
                borderRadius: '4px',
                color: '#A0AEC0'
              }}
            />
            <Line 
              type="monotone" 
              dataKey="NW" 
              stroke="#EF4444" 
              strokeWidth={2}
              dot={{ fill: '#EF4444' }}
            />
            <Line 
              type="monotone" 
              dataKey="NE" 
              stroke="#10B981" 
              strokeWidth={2}
              dot={{ fill: '#10B981' }}
            />
            <Line 
              type="monotone" 
              dataKey="SW" 
              stroke="#3B82F6" 
              strokeWidth={2}
              dot={{ fill: '#3B82F6' }}
            />
            <Line 
              type="monotone" 
              dataKey="SE" 
              stroke="#F59E0B" 
              strokeWidth={2}
              dot={{ fill: '#F59E0B' }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

const SectionControl = ({ sectionColors, setSectionColors }) => {
  return (
    <div className="w-full">
      <h2 className="text-xl font-bold mb-4">Section Control</h2>
      <div className="grid grid-cols-2 gap-2">
        {Object.entries(sectionColors).map(([section, color]) => (
          <button
            key={section}
            className="h-8 rounded flex items-center justify-center text-black font-medium"
            style={{ backgroundColor: color }}
            onClick={() => {
              const colors = [
                '#8B5CF6', // Purple
                '#3B82F6', // Blue
                '#10B981', // Green
                '#F59E0B', // Orange
                '#EC4899', // Pink
                '#6366F1', // Indigo
                '#14B8A6', // Teal
                '#EF4444', // Red
                '#EAB308', // Yellow
                '#4B5563'  // Gray
              ];
              const currentIndex = colors.indexOf(color);
              const nextColor = colors[(currentIndex + 1) % colors.length];
              setSectionColors(prev => ({ ...prev, [section]: nextColor }));
            }}
          >
            Section {section}
          </button>
        ))}
      </div>
    </div>
  );
};

const IntegratedArenaSystem = () => {
  const [devices, setDevices] = useState([]);
  const [wsConnected, setWsConnected] = useState(false);
  const [sectionColors, setSectionColors] = useState({
    'A': '#8B5CF6', // Purple
    'B': '#3B82F6', // Blue
    'C': '#10B981', // Green
    'D': '#F59E0B', // Orange
    'E': '#8B5CF6', // Purple
    'F': '#EC4899', // Pink
    'G': '#6366F1', // Indigo
    'H': '#14B8A6'  // Teal
  });

  const connectWebSocket = useCallback(() => {
    const ws = new WebSocket(`ws://${window.location.hostname}:8000/ws`);

    ws.onopen = () => {
      console.log('WebSocket Connected');
      setWsConnected(true);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'update' && data.devices) {
        setDevices(data.devices);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket Disconnected');
      setWsConnected(false);
      setTimeout(connectWebSocket, 2000);
    };

    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, []);

  useEffect(() => {
    const cleanup = connectWebSocket();
    return cleanup;
  }, [connectWebSocket]);

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <header className="mb-6">
        <h1 className="text-3xl font-bold mb-4">Arena Interactive System</h1>
        <div className="grid grid-cols-3 gap-4 mb-6">
          <div>
            <Radio className="w-6 h-6 mb-1" />
            <div>Pico Network</div>
          </div>
          <div>
            <Bluetooth className="w-6 h-6 mb-1" />
            <div>Bluetooth Scanner</div>
          </div>
          <div>
            <Wifi className="w-6 h-6 mb-1" />
            <div>Server Connection</div>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-2 gap-8">
        <div className="space-y-8">
          <DeviceList devices={devices} />
          <SignalStrengthChart devices={devices} />
        </div>
        
        <div className="space-y-8">
          <ArenaGrid devices={devices} sectionColors={sectionColors} />
          <SectionControl sectionColors={sectionColors} setSectionColors={setSectionColors} />
        </div>
      </div>
    </div>
  );
};

export default IntegratedArenaSystem;