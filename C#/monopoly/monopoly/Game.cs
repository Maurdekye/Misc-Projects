using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace monopoly
{
    class Game
    {
        public List<Property> PropertyList;
        public List<BoardSpace> GameBoard;
        public List<Player> Players;
        public Deck CommunityChest;
        public Deck Chance;

        private bool AnyHumanPlayers;

        public Game(List<Player> Players)
        {

        }

        public void LoadCommunityChestDeck()
        {
            CommunityChest = new Deck();
            CommunityChest.Enqueue(new Card(DeckType.CommunityChest, actor =>
            {
                actor.Tell("Advance Directly to Go");
                actor.MoveTo<GoSpace>();
            }));
            CommunityChest.Enqueue(new Card(DeckType.CommunityChest, actor =>
            {
                actor.Tell("Bank error in your favor - collect $200");
                actor.ModifyCapital(200);
            }));
            CommunityChest.Enqueue(new Card(DeckType.CommunityChest, actor =>
            {
                actor.Tell("Doctor's fee - pay $50");
                actor.Charge(50);
            }));
            CommunityChest.Enqueue(new GetOutOfJailCard(DeckType.CommunityChest, actor => actor.Tell("Get out of jail free; you can keep this card until you need it")));
            CommunityChest.Enqueue(new Card(DeckType.CommunityChest, actor =>
            {
                actor.Tell("Go directly to jail.");
                actor.MoveTo<JailSpace>();
            }));
            CommunityChest.Enqueue(new Card(DeckType.CommunityChest, actor =>
            {
                actor.Tell("Opera night; collect $50 from everyone else");
                actor.ModifyCapital(actor.PlayingGame.Players.Select(p => p == actor ? 0 : p.Charge(50)).Sum());
            }));
            CommunityChest.Enqueue(new Card(DeckType.CommunityChest, actor =>
            {
                actor.Tell("Holiday fund matures - collect $100");
                actor.ModifyCapital(100);
            }));
            CommunityChest.Enqueue(new Card(DeckType.CommunityChest, actor =>
            {
                actor.Tell("Income tax refund; collect $20");
                actor.ModifyCapital(20);
            }));
            CommunityChest.Enqueue(new Card(DeckType.CommunityChest, actor =>
            {
                actor.Tell("It's your birthday; collect $10 from all other players");
                actor.ModifyCapital(actor.PlayingGame.Players.Select(p => p == actor ? 0 : p.Charge(10)).Sum());
            }));
            CommunityChest.Enqueue(new Card(DeckType.CommunityChest, actor =>
            {
                actor.Tell("Life Insurance matures - collect $100");
                actor.ModifyCapital(100);
            }));
            CommunityChest.Enqueue(new Card(DeckType.CommunityChest, actor =>
            {
                actor.Tell("Play hostpital fees of $100");
                actor.Charge(100);
            }));
            CommunityChest.Enqueue(new Card(DeckType.CommunityChest, actor =>
            {
                actor.Tell("Play school fees of $150");
                actor.Charge(150);
            }));
            CommunityChest.Enqueue(new Card(DeckType.CommunityChest, actor =>
            {
                actor.Tell("Recieve $25 consultancy fee");
                actor.ModifyCapital(25);
            }));
            CommunityChest.Enqueue(new Card(DeckType.CommunityChest, actor =>
            {
                actor.Tell("You have won second place in a beauty contest; collect $10");
                actor.ModifyCapital(10);
            }));
            CommunityChest.Enqueue(new Card(DeckType.CommunityChest, actor =>
            {
                actor.Tell("You inherit $100");
                actor.ModifyCapital(100);
            }));
            CommunityChest.Enqueue(new Card(DeckType.CommunityChest, actor =>
            {
                int amount = Util.FilterSubclass<Property, ColoredProperty>(actor.GetProperties()).Select(p => p.Houses > 4 ? 115 : p.Houses * 40).Sum();
                if (amount > 0)
                {
                    actor.Tell($"You are assesed for street repairs, and require a payment of ${amount}");
                    actor.Charge(amount);
                }
                else
                {
                    actor.Tell("You are assesed for street repairs, but are not deemed applicable for payment. You pay nothing.");
                }
            }));
        }
        public void LoadChanceDeck()
        {
            Chance = new Deck();
            Chance.Enqueue(new Card(DeckType.Chance, actor =>
            {
                actor.Tell("Advance Directly to Go");
                actor.MoveTo<GoSpace>();
            }));
            Chance.Enqueue(new Card(DeckType.Chance, actor =>
            {
                actor.Tell("Advance to Illinois Ave");
                actor.MoveTo(s => s is PropertySpace && (s as PropertySpace).GetProperties().Single().Name == "Illinois Avenue");
                actor.PlayingGame.GameBoard[actor.BoardPosition].Land(actor);
            }));
            Chance.Enqueue(new Card(DeckType.Chance, actor =>
            {
                actor.Tell("Advance to St. Charles Place");
                actor.MoveTo(s => s is PropertySpace && (s as PropertySpace).GetProperties().Single().Name == "St. Charles Place");
                actor.PlayingGame.GameBoard[actor.BoardPosition].Land(actor);
            }));
            Chance.Enqueue(new Card(DeckType.Chance, actor =>
            {
                actor.Tell("Advance to the nearest utility and pay the owner 10 times the amount of a dice roll");
                actor.MoveTo(s => s is PropertySpace && (s as PropertySpace).GetProperties().Single() is UtilityProperty);
                UtilityProperty utility = (actor.PlayingGame.GameBoard[actor.BoardPosition] as PropertySpace).GetProperties().Single() as UtilityProperty;
                if (utility.Owner is Player)
                {
                    utility.Owner.ModifyCapital(actor.Charge(10 * Dice.Roll()));
                }
                else
                    actor.PlayingGame.GameBoard[actor.BoardPosition].Land(actor);
            }));
            Chance.Enqueue(new Card(DeckType.Chance, actor =>
            {
                actor.Tell("Advance token to the nearest Railroad and pay owner twice the rental to which he/she is otherwise entitled");
                actor.MoveTo(s => s is PropertySpace && (s as PropertySpace).GetProperties().Single() is StationProperty);
                StationProperty station = (actor.PlayingGame.GameBoard[actor.BoardPosition] as PropertySpace).GetProperties().Single() as StationProperty;
                if (station.Owner is Player)
                {
                    station.Owner.ModifyCapital(actor.Charge(2 * station.GetRent(actor)));
                }
                else
                    station.PlayingGame.GameBoard[actor.BoardPosition].Land(actor);

            }));
            Chance.Enqueue(new Card(DeckType.Chance, actor =>
            {
                actor.Tell("Bank pays you dividend of $50");
                actor.ModifyCapital(50);
            }));
            Chance.Enqueue(new GetOutOfJailCard(DeckType.Chance, actor => actor.Tell("Get out of Jail Free – This card may be kept until needed, or traded/sold")));
            Chance.Enqueue(new Card(DeckType.Chance, actor =>
            {
                actor.Tell("Go back 3 spaces");
                actor.BoardPosition = (actor.BoardPosition - 3) % actor.PlayingGame.GameBoard.Count;
                actor.PlayingGame.GameBoard[actor.BoardPosition].Land(actor);
            }));
            Chance.Enqueue(new Card(DeckType.Chance, actor =>
            {
                actor.Tell("Go directly to jail; do not pass go, do not collect $200");
                actor.SendToJail();
            }));
            Chance.Enqueue(new Card(DeckType.Chance, actor =>
            {
                int amount = Util.FilterSubclass<Property, ColoredProperty>(actor.GetProperties()).Select(p => p.Houses > 4 ? 100 : p.Houses * 25).Sum();
                if (amount > 0)
                {
                    actor.Tell($"You make general repairs on all of your properties, and end up paying ${amount}");
                    actor.Charge(amount);
                }
                else
                {
                    actor.Tell("You asses the need to make general repairs on any of your properties, but decide it is not necessary. You pay nothing.");
                }
            }));
            Chance.Enqueue(new Card(DeckType.Chance, actor =>
            {
                actor.Tell("Pay poor tax of $15");
                actor.Charge(15);
            }));
            Chance.Enqueue(new Card(DeckType.Chance, actor =>
            {
                actor.Tell("Take a trip to Reading Railroad");
                actor.MoveTo(s => s is PropertySpace && (s as PropertySpace).GetProperties().Single().Name == "Reading Railroad");
                actor.PlayingGame.GameBoard[actor.BoardPosition].Land(actor);
            }));
            Chance.Enqueue(new Card(DeckType.Chance, actor =>
            {
                actor.Tell("Take a walk on the Boardwalk - advance to Boardwalk");
                actor.MoveTo(s => s is PropertySpace && (s as PropertySpace).GetProperties().Single().Name == "Boardwalk");
                actor.PlayingGame.GameBoard[actor.BoardPosition].Land(actor);
            }));
            Chance.Enqueue(new Card(DeckType.Chance, actor =>
            {
                actor.Tell("You have been elected Chairman of the Board; you pay each other player $50");
                actor.PlayingGame.Players.Where(p => p != actor).ToList().ForEach(p => p.ModifyCapital(actor.Charge(50)));
            }));
            Chance.Enqueue(new Card(DeckType.Chance, actor =>
            {
                actor.Tell("Your building loan matures; collect $150");
                actor.ModifyCapital(150);
            }));
            Chance.Enqueue(new Card(DeckType.Chance, actor =>
            {
                actor.Tell("You have won a crossword competition; collect $100");
                actor.ModifyCapital(100);
            }));

        }
        public void CreateBoard()
        {
            GameBoard = new List<BoardSpace>
            {
                new GoSpace(),
                new PropertySpace(new ColoredProperty("Mediterranean Avenue", 60, 30, Color.Brown, new List<int> { 2, 10, 30, 90, 160, 250 }, 50)),
                new CommunityChestSpace(),
                new PropertySpace(new ColoredProperty("Baltic Avenue", 60, 30, Color.Brown, new List<int> { 4, 20, 60, 180, 320, 450 }, 50)),
                new IncomeTaxSpace(),
                new PropertySpace(new StationProperty("Reading Railroad", 200, 100)),
                new PropertySpace(new ColoredProperty("Oriental Avenue", 100, 50, Color.Blue, new List<int> { 6, 30, 90, 270, 400, 550 }, 50)),
                new ChanceSpace(),
                new PropertySpace(new ColoredProperty("Vermont Avenue", 100, 50, Color.Blue, new List<int> { 6, 30, 90, 270, 400, 550 }, 50)),
                new PropertySpace(new ColoredProperty("Conneticut Avenue", 120, 60, Color.Blue, new List<int> { 8, 40, 100, 300, 450, 600 }, 50)),
                new JailSpace(),
                new PropertySpace(new ColoredProperty("St. Charles Place", 140, 70, Color.Pink, new List<int> { 10, 50, 150, 450, 625, 750 }, 100)),
                new PropertySpace(new UtilityProperty("Electric Company", 150, 75)),
                new PropertySpace(new ColoredProperty("States Avenue", 140, 70, Color.Pink, new List<int> { 10, 50, 150, 450, 625, 750 }, 100)),
                new PropertySpace(new ColoredProperty("Virginia Avenue", 160, 80, Color.Pink, new List<int> { 12, 60, 180, 500, 700, 900 }, 100)),
                new PropertySpace(new StationProperty("Pennsylvania Railroad", 200, 100)),
                new PropertySpace(new ColoredProperty("St. James Place", 180, 90, Color.Orange, new List<int> { 14, 70, 200, 550, 750, 950 }, 100))
            };

        }

        public void RetakeCard(Card c)
        {
            if (c.FromDeck == DeckType.CommunityChest)
                CommunityChest.Enqueue(c);
            else if (c.FromDeck == DeckType.Chance)
                Chance.Enqueue(c);
        }

        public void Broadcast(string message)
        {
            if (AnyHumanPlayers)
            {
                Console.WriteLine(message);
                Console.ReadKey();
            }
        }

        public void AuctionOffProperty(Property property)
        {
            Broadcast($"We are about to begin the auction off of {property}");
            Dictionary<Player, bool> hasPassed = new Dictionary<Player, bool>();
            Players.ForEach(p => hasPassed.Add(p, false));
            int auctionValue = property.Price / 2;
            while (hasPassed.Where(kv => kv.Value).Count() > 1)
            {
                foreach (Player player in Players)
                {
                    if (hasPassed[player])
                        continue;
                    AuctionResponse response = player.AskAuctionParticipation(property, auctionValue);
                    if (response is AuctionBuy)
                    {
                        int amount = (response as AuctionBuy).Amount;
                        auctionValue += amount;
                        Broadcast($"{player} has raised the price by ${amount}, it is now ${auctionValue}.");
                    }
                    else if (response is AuctionPass)
                    {
                        Broadcast($"{player} has passed.");
                        hasPassed[player] = true;
                    }
                }
            }
            if (hasPassed.Where(kv => kv.Value).Count() == 1)
            {
                Player winner = hasPassed.Where(kv => kv.Value).Single().Key;
                Broadcast($"{winner} has won the auction, and purchases {property} at a price of ${auctionValue}");
                winner.Charge(auctionValue);
                property.Owner = winner;
            }
            else
            {
                Broadcast($"All players simultaneously passed, {property} goes unbought.");
            }
        }
    }
}
