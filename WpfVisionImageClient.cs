using System;
using System.Buffers.Binary;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;

namespace VisionClient
{
    // Read-only image client. It never controls a camera. Detection triggers still
    // flow WPF -> PLC (ADS) -> Python Vision Service -> Camera Manager.
    public sealed class VisionImageClient : IAsyncDisposable
    {
        private readonly TcpClient _client = new();
        private NetworkStream? _stream;

        public async Task ConnectAsync(string host, int port = 50010, CancellationToken ct = default)
        {
            await _client.ConnectAsync(host, port, ct);
            _stream = _client.GetStream();
        }

        public async Task<(JsonDocument Metadata, BitmapImage? Image)> GetLatestAsync(
            string channel, CancellationToken ct = default)
        {
            if (_stream == null) throw new InvalidOperationException("Not connected");
            if (channel != "screw" && channel != "coax") throw new ArgumentOutOfRangeException(nameof(channel));

            byte[] request = Encoding.UTF8.GetBytes($"GET {channel}\n");
            await _stream.WriteAsync(request, ct);

            int headerLength = await ReadInt32BEAsync(_stream, ct);
            byte[] header = await ReadExactAsync(_stream, headerLength, ct);
            int jpegLength = await ReadInt32BEAsync(_stream, ct);
            byte[] jpeg = await ReadExactAsync(_stream, jpegLength, ct);

            JsonDocument metadata = JsonDocument.Parse(header);
            if (jpeg.Length == 0) return (metadata, null);

            BitmapImage bitmap = new();
            using MemoryStream ms = new(jpeg, writable: false);
            bitmap.BeginInit();
            bitmap.CacheOption = BitmapCacheOption.OnLoad;
            bitmap.StreamSource = ms;
            bitmap.EndInit();
            bitmap.Freeze();
            return (metadata, bitmap);
        }

        private static async Task<int> ReadInt32BEAsync(Stream stream, CancellationToken ct)
        {
            byte[] b = await ReadExactAsync(stream, 4, ct);
            return BinaryPrimitives.ReadInt32BigEndian(b);
        }

        private static async Task<byte[]> ReadExactAsync(Stream stream, int length, CancellationToken ct)
        {
            byte[] data = new byte[length];
            int offset = 0;
            while (offset < length)
            {
                int n = await stream.ReadAsync(data.AsMemory(offset, length - offset), ct);
                if (n == 0) throw new EndOfStreamException();
                offset += n;
            }
            return data;
        }

        public async ValueTask DisposeAsync()
        {
            if (_stream != null) await _stream.DisposeAsync();
            _client.Dispose();
        }
    }
}
