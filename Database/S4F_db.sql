USE [master]
GO
/****** Object:  Database [S4F_clone]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE DATABASE [S4F_clone]
 CONTAINMENT = NONE
 ON  PRIMARY 
( NAME = N'S4F', FILENAME = N'D:\SqlData\S4F_clone_clone.mdf' , SIZE = 8192KB , MAXSIZE = UNLIMITED, FILEGROWTH = 65536KB )
 LOG ON 
( NAME = N'S4F_log', FILENAME = N'D:\SqlData\S4F_clone_clone_log.ldf' , SIZE = 8192KB , MAXSIZE = 2048GB , FILEGROWTH = 65536KB )
 WITH CATALOG_COLLATION = DATABASE_DEFAULT
GO
ALTER DATABASE [S4F_clone] SET COMPATIBILITY_LEVEL = 150
GO
IF (1 = FULLTEXTSERVICEPROPERTY('IsFullTextInstalled'))
begin
EXEC [S4F_clone].[dbo].[sp_fulltext_database] @action = 'enable'
end
GO
ALTER DATABASE [S4F_clone] SET ANSI_NULL_DEFAULT OFF 
GO
ALTER DATABASE [S4F_clone] SET ANSI_NULLS OFF 
GO
ALTER DATABASE [S4F_clone] SET ANSI_PADDING OFF 
GO
ALTER DATABASE [S4F_clone] SET ANSI_WARNINGS OFF 
GO
ALTER DATABASE [S4F_clone] SET ARITHABORT OFF 
GO
ALTER DATABASE [S4F_clone] SET AUTO_CLOSE OFF 
GO
ALTER DATABASE [S4F_clone] SET AUTO_SHRINK OFF 
GO
ALTER DATABASE [S4F_clone] SET AUTO_UPDATE_STATISTICS ON 
GO
ALTER DATABASE [S4F_clone] SET CURSOR_CLOSE_ON_COMMIT OFF 
GO
ALTER DATABASE [S4F_clone] SET CURSOR_DEFAULT  GLOBAL 
GO
ALTER DATABASE [S4F_clone] SET CONCAT_NULL_YIELDS_NULL OFF 
GO
ALTER DATABASE [S4F_clone] SET NUMERIC_ROUNDABORT OFF 
GO
ALTER DATABASE [S4F_clone] SET QUOTED_IDENTIFIER OFF 
GO
ALTER DATABASE [S4F_clone] SET RECURSIVE_TRIGGERS OFF 
GO
ALTER DATABASE [S4F_clone] SET  DISABLE_BROKER 
GO
ALTER DATABASE [S4F_clone] SET AUTO_UPDATE_STATISTICS_ASYNC OFF 
GO
ALTER DATABASE [S4F_clone] SET DATE_CORRELATION_OPTIMIZATION OFF 
GO
ALTER DATABASE [S4F_clone] SET TRUSTWORTHY OFF 
GO
ALTER DATABASE [S4F_clone] SET ALLOW_SNAPSHOT_ISOLATION OFF 
GO
ALTER DATABASE [S4F_clone] SET PARAMETERIZATION SIMPLE 
GO
ALTER DATABASE [S4F_clone] SET READ_COMMITTED_SNAPSHOT OFF 
GO
ALTER DATABASE [S4F_clone] SET HONOR_BROKER_PRIORITY OFF 
GO
ALTER DATABASE [S4F_clone] SET RECOVERY FULL 
GO
ALTER DATABASE [S4F_clone] SET  MULTI_USER 
GO
ALTER DATABASE [S4F_clone] SET PAGE_VERIFY CHECKSUM  
GO
ALTER DATABASE [S4F_clone] SET DB_CHAINING OFF 
GO
ALTER DATABASE [S4F_clone] SET FILESTREAM( NON_TRANSACTED_ACCESS = OFF ) 
GO
ALTER DATABASE [S4F_clone] SET TARGET_RECOVERY_TIME = 60 SECONDS 
GO
ALTER DATABASE [S4F_clone] SET DELAYED_DURABILITY = DISABLED 
GO
EXEC sys.sp_db_vardecimal_storage_format N'S4F_clone', N'ON'
GO
ALTER DATABASE [S4F_clone] SET QUERY_STORE = OFF
GO
USE [S4F_clone]
GO
/****** Object:  Table [dbo].[Adapter]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Adapter](
	[Oid] [int] IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
	[CreatedBy] [uniqueidentifier] NULL,
	[UpdatedBy] [uniqueidentifier] NULL,
	[CreatedOn] [smalldatetime] NOT NULL,
	[UpdatedOn] [smalldatetime] NOT NULL,
	[Description] [nvarchar](100) NULL,
	[AlternativeDescription] [nvarchar](100) NULL,
	[Active] [bit] NOT NULL,
	[Ordering] [int] NOT NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_Adapter] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Analysis]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Analysis](
	[Oid] [uniqueidentifier] ROWGUIDCOL  NOT NULL,
	[DimensionPropertiesString] [nvarchar](max) NULL,
	[Name] [nvarchar](100) NULL,
	[Criteria] [nvarchar](max) NULL,
	[ObjectTypeName] [nvarchar](max) NULL,
	[ChartSettingsContent] [varbinary](max) NULL,
	[PivotGridSettingsContent] [varbinary](max) NULL,
	[OptimisticLockField] [int] NULL,
	[GCRecord] [int] NULL,
 CONSTRAINT [PK_Analysis] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[AuditDataItemPersistent]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[AuditDataItemPersistent](
	[Oid] [uniqueidentifier] ROWGUIDCOL  NOT NULL,
	[UserName] [nvarchar](100) NULL,
	[ModifiedOn] [datetime] NULL,
	[OperationType] [nvarchar](100) NULL,
	[Description] [nvarchar](2048) NULL,
	[AuditedObject] [uniqueidentifier] NULL,
	[OldObject] [uniqueidentifier] NULL,
	[NewObject] [uniqueidentifier] NULL,
	[OldValue] [nvarchar](1024) NULL,
	[NewValue] [nvarchar](1024) NULL,
	[PropertyName] [nvarchar](100) NULL,
	[OptimisticLockField] [int] NULL,
	[GCRecord] [int] NULL,
 CONSTRAINT [PK_AuditDataItemPersistent] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[AuditedObjectWeakReference]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[AuditedObjectWeakReference](
	[Oid] [uniqueidentifier] ROWGUIDCOL  NOT NULL,
	[GuidId] [uniqueidentifier] NULL,
	[IntId] [int] NULL,
	[DisplayName] [nvarchar](250) NULL,
 CONSTRAINT [PK_AuditedObjectWeakReference] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Brand]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Brand](
	[Oid] [int] IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
	[CreatedBy] [uniqueidentifier] NULL,
	[UpdatedBy] [uniqueidentifier] NULL,
	[CreatedOn] [smalldatetime] NOT NULL,
	[UpdatedOn] [smalldatetime] NOT NULL,
	[Description] [nvarchar](100) NULL,
	[AlternativeDescription] [nvarchar](100) NULL,
	[Active] [bit] NOT NULL,
	[Ordering] [int] NOT NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_Brand] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[BusinessUnit]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[BusinessUnit](
	[Oid] [int] IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
	[CreatedBy] [uniqueidentifier] NULL,
	[UpdatedBy] [uniqueidentifier] NULL,
	[CreatedOn] [smalldatetime] NOT NULL,
	[UpdatedOn] [smalldatetime] NOT NULL,
	[Description] [nvarchar](100) NULL,
	[AlternativeDescription] [nvarchar](100) NULL,
	[Active] [bit] NOT NULL,
	[Ordering] [int] NOT NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_BusinessUnit] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Cluster]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Cluster](
	[Oid] [int] IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
	[CreatedBy] [uniqueidentifier] NULL,
	[UpdatedBy] [uniqueidentifier] NULL,
	[CreatedOn] [smalldatetime] NOT NULL,
	[UpdatedOn] [smalldatetime] NOT NULL,
	[Description] [nvarchar](100) NULL,
	[AlternativeDescription] [nvarchar](100) NULL,
	[Active] [bit] NOT NULL,
	[Ordering] [int] NOT NULL,
	[ProductCategory] [int] NULL,
	[ProductSubcategory] [int] NULL,
	[Gender] [int] NULL,
	[Lifestage] [int] NULL,
	[Length] [int] NULL,
	[Sleeve] [int] NULL,
	[CollarDesign] [int] NULL,
	[NeckDesign] [int] NULL,
	[Fit] [int] NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_Cluster] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[CollarDesign]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[CollarDesign](
	[Oid] [int] IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
	[CreatedBy] [uniqueidentifier] NULL,
	[UpdatedBy] [uniqueidentifier] NULL,
	[CreatedOn] [smalldatetime] NOT NULL,
	[UpdatedOn] [smalldatetime] NOT NULL,
	[Description] [nvarchar](100) NULL,
	[AlternativeDescription] [nvarchar](100) NULL,
	[Active] [bit] NOT NULL,
	[Ordering] [int] NOT NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_CollarDesign] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[ColorRGB]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[ColorRGB](
	[Oid] [int] IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
	[CreatedBy] [uniqueidentifier] NULL,
	[UpdatedBy] [uniqueidentifier] NULL,
	[CreatedOn] [smalldatetime] NOT NULL,
	[UpdatedOn] [smalldatetime] NOT NULL,
	[Description] [nvarchar](100) NULL,
	[AlternativeDescription] [nvarchar](100) NULL,
	[Active] [bit] NOT NULL,
	[Ordering] [int] NOT NULL,
	[RGBcolor] [int] NULL,
	[Red] [int] NULL,
	[Green] [int] NULL,
	[Blue] [int] NULL,
	[Label] [nvarchar](100) NULL,
	[LabelDetailed] [nvarchar](200) NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_ColorRGB] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Combination]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Combination](
	[Oid] [int] IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
	[CreatedBy] [uniqueidentifier] NULL,
	[UpdatedBy] [uniqueidentifier] NULL,
	[CreatedOn] [smalldatetime] NOT NULL,
	[UpdatedOn] [smalldatetime] NOT NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_Combination] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[CrawlSearch]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[CrawlSearch](
	[Oid] [int] IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
	[CreatedBy] [uniqueidentifier] NULL,
	[UpdatedBy] [uniqueidentifier] NULL,
	[CreatedOn] [smalldatetime] NOT NULL,
	[UpdatedOn] [smalldatetime] NOT NULL,
	[Description] [nvarchar](100) NULL,
	[Adapter] [int] NULL,
	[SearchTerm] [nvarchar](100) NULL,
	[NumberOfProductsToReturn] [int] NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_CrawlSearch] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Dashboard]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Dashboard](
	[Oid] [int] IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
	[CreatedBy] [uniqueidentifier] NULL,
	[UpdatedBy] [uniqueidentifier] NULL,
	[CreatedOn] [smalldatetime] NOT NULL,
	[UpdatedOn] [smalldatetime] NOT NULL,
	[Description] [nvarchar](100) NULL,
	[AlternativeDescription] [nvarchar](100) NULL,
	[Active] [bit] NOT NULL,
	[Ordering] [int] NOT NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_Dashboard] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[DashboardData]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[DashboardData](
	[Oid] [uniqueidentifier] ROWGUIDCOL  NOT NULL,
	[Content] [nvarchar](max) NULL,
	[Title] [nvarchar](100) NULL,
	[SynchronizeTitle] [bit] NULL,
	[OptimisticLockField] [int] NULL,
	[GCRecord] [int] NULL,
 CONSTRAINT [PK_DashboardData] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[DashboardProduct]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[DashboardProduct](
	[Oid] [int] IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
	[CreatedBy] [uniqueidentifier] NULL,
	[UpdatedBy] [uniqueidentifier] NULL,
	[CreatedOn] [smalldatetime] NOT NULL,
	[UpdatedOn] [smalldatetime] NOT NULL,
	[Dashboard] [int] NULL,
	[Product] [int] NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_DashboardProduct] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Event]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Event](
	[Oid] [uniqueidentifier] ROWGUIDCOL  NOT NULL,
	[ResourceIds] [nvarchar](max) NULL,
	[RecurrencePattern] [uniqueidentifier] NULL,
	[Subject] [nvarchar](250) NULL,
	[Description] [nvarchar](max) NULL,
	[StartOn] [datetime] NULL,
	[EndOn] [datetime] NULL,
	[AllDay] [bit] NULL,
	[Location] [nvarchar](100) NULL,
	[Label] [int] NULL,
	[Status] [int] NULL,
	[Type] [int] NULL,
	[RemindIn] [float] NULL,
	[ReminderInfoXml] [nvarchar](200) NULL,
	[AlarmTime] [datetime] NULL,
	[IsPostponed] [bit] NULL,
	[RecurrenceInfoXml] [nvarchar](max) NULL,
	[OptimisticLockField] [int] NULL,
	[GCRecord] [int] NULL,
 CONSTRAINT [PK_Event] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[FileData]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[FileData](
	[Oid] [uniqueidentifier] ROWGUIDCOL  NOT NULL,
	[size] [int] NULL,
	[FileName] [nvarchar](260) NULL,
	[Content] [varbinary](max) NULL,
	[OptimisticLockField] [int] NULL,
	[GCRecord] [int] NULL,
 CONSTRAINT [PK_FileData] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[FilteringCriterion]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[FilteringCriterion](
	[Oid] [uniqueidentifier] ROWGUIDCOL  NOT NULL,
	[Description] [nvarchar](100) NULL,
	[ObjectType] [nvarchar](100) NULL,
	[Criterion] [nvarchar](max) NULL,
	[OptimisticLockField] [int] NULL,
	[GCRecord] [int] NULL,
 CONSTRAINT [PK_FilteringCriterion] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Fit]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Fit](
	[Oid] [int] IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
	[CreatedBy] [uniqueidentifier] NULL,
	[UpdatedBy] [uniqueidentifier] NULL,
	[CreatedOn] [smalldatetime] NOT NULL,
	[UpdatedOn] [smalldatetime] NOT NULL,
	[Description] [nvarchar](100) NULL,
	[AlternativeDescription] [nvarchar](100) NULL,
	[Active] [bit] NOT NULL,
	[Ordering] [int] NOT NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_Fit] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Gender]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Gender](
	[Oid] [int] IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
	[CreatedBy] [uniqueidentifier] NULL,
	[UpdatedBy] [uniqueidentifier] NULL,
	[CreatedOn] [smalldatetime] NOT NULL,
	[UpdatedOn] [smalldatetime] NOT NULL,
	[Description] [nvarchar](100) NULL,
	[AlternativeDescription] [nvarchar](100) NULL,
	[Active] [bit] NOT NULL,
	[Ordering] [int] NOT NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_Gender] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[HCategory]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[HCategory](
	[Oid] [uniqueidentifier] ROWGUIDCOL  NOT NULL,
	[Parent] [uniqueidentifier] NULL,
	[Name] [nvarchar](100) NULL,
	[OptimisticLockField] [int] NULL,
	[GCRecord] [int] NULL,
 CONSTRAINT [PK_HCategory] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[InspirationBackground]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[InspirationBackground](
	[Oid] [int] IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
	[CreatedBy] [uniqueidentifier] NULL,
	[UpdatedBy] [uniqueidentifier] NULL,
	[CreatedOn] [smalldatetime] NOT NULL,
	[UpdatedOn] [smalldatetime] NOT NULL,
	[Description] [nvarchar](100) NULL,
	[AlternativeDescription] [nvarchar](100) NULL,
	[Active] [bit] NOT NULL,
	[Ordering] [int] NOT NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_InspirationBackground] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[KpiDefinition]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[KpiDefinition](
	[Oid] [uniqueidentifier] ROWGUIDCOL  NOT NULL,
	[TargetObjectType] [nvarchar](max) NULL,
	[Changed] [datetime] NULL,
	[KpiInstance] [uniqueidentifier] NULL,
	[Name] [nvarchar](100) NULL,
	[Active] [bit] NULL,
	[Criteria] [nvarchar](max) NULL,
	[Expression] [nvarchar](max) NULL,
	[GreenZone] [float] NULL,
	[RedZone] [float] NULL,
	[Range] [nvarchar](100) NULL,
	[Compare] [bit] NULL,
	[RangeToCompare] [nvarchar](100) NULL,
	[MeasurementFrequency] [int] NULL,
	[MeasurementMode] [int] NULL,
	[Direction] [int] NULL,
	[ChangedOn] [datetime] NULL,
	[SuppressedSeries] [nvarchar](100) NULL,
	[EnableCustomizeRepresentation] [bit] NULL,
	[OptimisticLockField] [int] NULL,
	[GCRecord] [int] NULL,
 CONSTRAINT [PK_KpiDefinition] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[KpiHistoryItem]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[KpiHistoryItem](
	[Oid] [uniqueidentifier] ROWGUIDCOL  NOT NULL,
	[KpiInstance] [uniqueidentifier] NULL,
	[RangeStart] [datetime] NULL,
	[RangeEnd] [datetime] NULL,
	[Value] [float] NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_KpiHistoryItem] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[KpiInstance]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[KpiInstance](
	[Oid] [uniqueidentifier] ROWGUIDCOL  NOT NULL,
	[ForceMeasurementDateTime] [datetime] NULL,
	[KpiDefinition] [uniqueidentifier] NULL,
	[Settings] [nvarchar](max) NULL,
	[OptimisticLockField] [int] NULL,
	[GCRecord] [int] NULL,
 CONSTRAINT [PK_KpiInstance] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[KpiScorecard]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[KpiScorecard](
	[Oid] [uniqueidentifier] ROWGUIDCOL  NOT NULL,
	[Name] [nvarchar](100) NULL,
	[OptimisticLockField] [int] NULL,
	[GCRecord] [int] NULL,
 CONSTRAINT [PK_KpiScorecard] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[KpiScorecardScorecards_KpiInstanceIndicators]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[KpiScorecardScorecards_KpiInstanceIndicators](
	[Indicators] [uniqueidentifier] NULL,
	[Scorecards] [uniqueidentifier] NULL,
	[OID] [uniqueidentifier] ROWGUIDCOL  NOT NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_KpiScorecardScorecards_KpiInstanceIndicators] PRIMARY KEY CLUSTERED 
(
	[OID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Length]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Length](
	[Oid] [int] IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
	[CreatedBy] [uniqueidentifier] NULL,
	[UpdatedBy] [uniqueidentifier] NULL,
	[CreatedOn] [smalldatetime] NOT NULL,
	[UpdatedOn] [smalldatetime] NOT NULL,
	[Description] [nvarchar](100) NULL,
	[AlternativeDescription] [nvarchar](100) NULL,
	[Active] [bit] NOT NULL,
	[Ordering] [int] NOT NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_Length] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[LifeStage]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[LifeStage](
	[Oid] [int] IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
	[CreatedBy] [uniqueidentifier] NULL,
	[UpdatedBy] [uniqueidentifier] NULL,
	[CreatedOn] [smalldatetime] NOT NULL,
	[UpdatedOn] [smalldatetime] NOT NULL,
	[Description] [nvarchar](100) NULL,
	[AlternativeDescription] [nvarchar](100) NULL,
	[Active] [bit] NOT NULL,
	[Ordering] [int] NOT NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_LifeStage] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Manufacturer]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Manufacturer](
	[Oid] [int] IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
	[CreatedBy] [uniqueidentifier] NULL,
	[UpdatedBy] [uniqueidentifier] NULL,
	[CreatedOn] [smalldatetime] NOT NULL,
	[UpdatedOn] [smalldatetime] NOT NULL,
	[Description] [nvarchar](100) NULL,
	[AlternativeDescription] [nvarchar](100) NULL,
	[Active] [bit] NOT NULL,
	[Ordering] [int] NOT NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_Manufacturer] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[ModelDifference]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[ModelDifference](
	[Oid] [uniqueidentifier] ROWGUIDCOL  NOT NULL,
	[UserId] [nvarchar](100) NULL,
	[ContextId] [nvarchar](100) NULL,
	[Version] [int] NULL,
	[OptimisticLockField] [int] NULL,
	[GCRecord] [int] NULL,
 CONSTRAINT [PK_ModelDifference] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[ModelDifferenceAspect]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[ModelDifferenceAspect](
	[Oid] [uniqueidentifier] ROWGUIDCOL  NOT NULL,
	[Name] [nvarchar](100) NULL,
	[Xml] [nvarchar](max) NULL,
	[Owner] [uniqueidentifier] NULL,
	[OptimisticLockField] [int] NULL,
	[GCRecord] [int] NULL,
 CONSTRAINT [PK_ModelDifferenceAspect] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[NeckDesign]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[NeckDesign](
	[Oid] [int] IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
	[CreatedBy] [uniqueidentifier] NULL,
	[UpdatedBy] [uniqueidentifier] NULL,
	[CreatedOn] [smalldatetime] NOT NULL,
	[UpdatedOn] [smalldatetime] NOT NULL,
	[Description] [nvarchar](100) NULL,
	[AlternativeDescription] [nvarchar](100) NULL,
	[Active] [bit] NOT NULL,
	[Ordering] [int] NOT NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_NeckDesign] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[PermissionPolicyActionPermissionObject]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[PermissionPolicyActionPermissionObject](
	[Oid] [uniqueidentifier] ROWGUIDCOL  NOT NULL,
	[ActionId] [nvarchar](100) NULL,
	[Role] [uniqueidentifier] NULL,
	[OptimisticLockField] [int] NULL,
	[GCRecord] [int] NULL,
 CONSTRAINT [PK_PermissionPolicyActionPermissionObject] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[PermissionPolicyMemberPermissionsObject]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[PermissionPolicyMemberPermissionsObject](
	[Oid] [uniqueidentifier] ROWGUIDCOL  NOT NULL,
	[Members] [nvarchar](max) NULL,
	[ReadState] [int] NULL,
	[WriteState] [int] NULL,
	[Criteria] [nvarchar](max) NULL,
	[TypePermissionObject] [uniqueidentifier] NULL,
	[OptimisticLockField] [int] NULL,
	[GCRecord] [int] NULL,
 CONSTRAINT [PK_PermissionPolicyMemberPermissionsObject] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[PermissionPolicyNavigationPermissionsObject]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[PermissionPolicyNavigationPermissionsObject](
	[Oid] [uniqueidentifier] ROWGUIDCOL  NOT NULL,
	[ItemPath] [nvarchar](max) NULL,
	[NavigateState] [int] NULL,
	[Role] [uniqueidentifier] NULL,
	[OptimisticLockField] [int] NULL,
	[GCRecord] [int] NULL,
 CONSTRAINT [PK_PermissionPolicyNavigationPermissionsObject] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[PermissionPolicyObjectPermissionsObject]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[PermissionPolicyObjectPermissionsObject](
	[Oid] [uniqueidentifier] ROWGUIDCOL  NOT NULL,
	[Criteria] [nvarchar](max) NULL,
	[ReadState] [int] NULL,
	[WriteState] [int] NULL,
	[DeleteState] [int] NULL,
	[NavigateState] [int] NULL,
	[TypePermissionObject] [uniqueidentifier] NULL,
	[OptimisticLockField] [int] NULL,
	[GCRecord] [int] NULL,
 CONSTRAINT [PK_PermissionPolicyObjectPermissionsObject] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[PermissionPolicyRole]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[PermissionPolicyRole](
	[Oid] [uniqueidentifier] ROWGUIDCOL  NOT NULL,
	[Name] [nvarchar](100) NULL,
	[IsAdministrative] [bit] NULL,
	[CanEditModel] [bit] NULL,
	[PermissionPolicy] [int] NULL,
	[OptimisticLockField] [int] NULL,
	[GCRecord] [int] NULL,
	[ObjectType] [int] NULL,
 CONSTRAINT [PK_PermissionPolicyRole] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[PermissionPolicyTypePermissionsObject]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[PermissionPolicyTypePermissionsObject](
	[Oid] [uniqueidentifier] ROWGUIDCOL  NOT NULL,
	[Role] [uniqueidentifier] NULL,
	[TargetType] [nvarchar](max) NULL,
	[ReadState] [int] NULL,
	[WriteState] [int] NULL,
	[CreateState] [int] NULL,
	[DeleteState] [int] NULL,
	[NavigateState] [int] NULL,
	[OptimisticLockField] [int] NULL,
	[GCRecord] [int] NULL,
 CONSTRAINT [PK_PermissionPolicyTypePermissionsObject] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[PermissionPolicyUser]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[PermissionPolicyUser](
	[Oid] [uniqueidentifier] ROWGUIDCOL  NOT NULL,
	[StoredPassword] [nvarchar](max) NULL,
	[ChangePasswordOnFirstLogon] [bit] NULL,
	[UserName] [nvarchar](100) NULL,
	[IsActive] [bit] NULL,
	[OptimisticLockField] [int] NULL,
	[GCRecord] [int] NULL,
 CONSTRAINT [PK_PermissionPolicyUser] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[PermissionPolicyUserUsers_PermissionPolicyRoleRoles]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[PermissionPolicyUserUsers_PermissionPolicyRoleRoles](
	[Roles] [uniqueidentifier] NULL,
	[Users] [uniqueidentifier] NULL,
	[OID] [uniqueidentifier] ROWGUIDCOL  NOT NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_PermissionPolicyUserUsers_PermissionPolicyRoleRoles] PRIMARY KEY CLUSTERED 
(
	[OID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Product]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Product](
	[Oid] [int] IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
	[CreatedBy] [uniqueidentifier] NULL,
	[UpdatedBy] [uniqueidentifier] NULL,
	[CreatedOn] [smalldatetime] NOT NULL,
	[UpdatedOn] [smalldatetime] NOT NULL,
	[Description] [nvarchar](100) NULL,
	[AlternativeDescription] [nvarchar](100) NULL,
	[Active] [bit] NOT NULL,
	[Ordering] [int] NOT NULL,
	[ProductCode] [nvarchar](40) NULL,
	[ProductTitle] [nvarchar](512) NULL,
	[Composition] [nvarchar](2000) NULL,
	[ForeignComposition] [nvarchar](2000) NULL,
	[SiteHeadline] [nvarchar](500) NULL,
	[ColorsDescription] [nvarchar](max) NULL,
	[Metadata] [nvarchar](2000) NULL,
	[SamplePrice] [decimal](9, 2) NULL,
	[ProductionPrice] [decimal](9, 2) NULL,
	[WholesalePrice] [decimal](9, 2) NULL,
	[RetailPrice] [decimal](9, 2) NULL,
	[Image] [varbinary](max) NULL,
	[Photo] [nvarchar](254) NULL,
	[Sketch] [nvarchar](254) NULL,
	[URL] [nvarchar](100) NULL,
	[ImageSource] [nvarchar](500) NULL,
	[Adapter] [int] NULL,
	[Brand] [int] NULL,
	[Fit] [int] NULL,
	[CollarDesign] [int] NULL,
	[SampleManufacturer] [int] NULL,
	[ProductionManufacturer] [int] NULL,
	[Length] [int] NULL,
	[NeckDesign] [int] NULL,
	[ProductCategory] [int] NULL,
	[ProductSubcategory] [int] NULL,
	[Sleeve] [int] NULL,
	[LifeStage] [int] NULL,
	[TrendTheme] [int] NULL,
	[InspirationBackground] [int] NULL,
	[Gender] [int] NULL,
	[BusinessUnit] [int] NULL,
	[Season] [int] NULL,
	[Cluster] [int] NULL,
	[FinancialCluster] [int] NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_Product] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[ProductCategory]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[ProductCategory](
	[Oid] [int] IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
	[CreatedBy] [uniqueidentifier] NULL,
	[UpdatedBy] [uniqueidentifier] NULL,
	[CreatedOn] [smalldatetime] NOT NULL,
	[UpdatedOn] [smalldatetime] NOT NULL,
	[Description] [nvarchar](100) NULL,
	[AlternativeDescription] [nvarchar](100) NULL,
	[Active] [bit] NOT NULL,
	[Ordering] [int] NOT NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_ProductCategory] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[ProductColor]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[ProductColor](
	[Oid] [int] IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
	[CreatedBy] [uniqueidentifier] NULL,
	[UpdatedBy] [uniqueidentifier] NULL,
	[CreatedOn] [smalldatetime] NOT NULL,
	[UpdatedOn] [smalldatetime] NOT NULL,
	[Product] [int] NULL,
	[ColorRGB] [int] NULL,
	[Ranking] [int] NULL,
	[Percentage] [int] NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_ProductColor] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[ProductHistory]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[ProductHistory](
	[Oid] [int] IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
	[CreatedBy] [uniqueidentifier] NULL,
	[UpdatedBy] [uniqueidentifier] NULL,
	[CreatedOn] [smalldatetime] NOT NULL,
	[UpdatedOn] [smalldatetime] NOT NULL,
	[Product] [int] NULL,
	[SearchDate] [datetime] NULL,
	[ReferenceOrder] [int] NULL,
	[TrendingOrder] [int] NULL,
	[Price] [decimal](9, 2) NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_ProductHistory] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[ProductSubcategory]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[ProductSubcategory](
	[Oid] [int] IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
	[CreatedBy] [uniqueidentifier] NULL,
	[UpdatedBy] [uniqueidentifier] NULL,
	[CreatedOn] [smalldatetime] NOT NULL,
	[UpdatedOn] [smalldatetime] NOT NULL,
	[Description] [nvarchar](100) NULL,
	[AlternativeDescription] [nvarchar](100) NULL,
	[Active] [bit] NOT NULL,
	[Ordering] [int] NOT NULL,
	[ProductCategory] [int] NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_ProductSubcategory] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[ReportDataV2]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[ReportDataV2](
	[Oid] [uniqueidentifier] ROWGUIDCOL  NOT NULL,
	[ObjectTypeName] [nvarchar](512) NULL,
	[Content] [varbinary](max) NULL,
	[Name] [nvarchar](100) NULL,
	[ParametersObjectTypeName] [nvarchar](512) NULL,
	[IsInplaceReport] [bit] NULL,
	[PredefinedReportType] [nvarchar](512) NULL,
	[OptimisticLockField] [int] NULL,
	[GCRecord] [int] NULL,
 CONSTRAINT [PK_ReportDataV2] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Resource]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Resource](
	[Oid] [uniqueidentifier] ROWGUIDCOL  NOT NULL,
	[Caption] [nvarchar](100) NULL,
	[Color] [int] NULL,
	[OptimisticLockField] [int] NULL,
	[GCRecord] [int] NULL,
 CONSTRAINT [PK_Resource] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[ResourceResources_EventEvents]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[ResourceResources_EventEvents](
	[Events] [uniqueidentifier] NULL,
	[Resources] [uniqueidentifier] NULL,
	[OID] [uniqueidentifier] ROWGUIDCOL  NOT NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_ResourceResources_EventEvents] PRIMARY KEY CLUSTERED 
(
	[OID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Restriction]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Restriction](
	[Oid] [int] IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
	[CreatedBy] [uniqueidentifier] NULL,
	[UpdatedBy] [uniqueidentifier] NULL,
	[CreatedOn] [smalldatetime] NOT NULL,
	[UpdatedOn] [smalldatetime] NOT NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_Restriction] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Result]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Result](
	[Oid] [int] IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
	[CreatedBy] [uniqueidentifier] NULL,
	[UpdatedBy] [uniqueidentifier] NULL,
	[CreatedOn] [smalldatetime] NOT NULL,
	[UpdatedOn] [smalldatetime] NOT NULL,
	[Search] [int] NULL,
	[Product] [int] NULL,
	[Clicked] [bit] NOT NULL,
	[IsFavorite] [bit] NOT NULL,
	[IsIrrelevant] [bit] NOT NULL,
	[GradeBySystem] [int] NULL,
	[GradeByUser] [int] NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_Result] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Search]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Search](
	[Oid] [int] IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
	[CreatedBy] [uniqueidentifier] NULL,
	[UpdatedBy] [uniqueidentifier] NULL,
	[CreatedOn] [smalldatetime] NOT NULL,
	[UpdatedOn] [smalldatetime] NOT NULL,
	[Criteria] [nvarchar](max) NULL,
	[Round] [int] NULL,
	[Season] [int] NULL,
	[ProductsAfter] [datetime] NULL,
	[NumberOfProductsToReturn] [int] NULL,
	[TrendPercentage] [int] NULL,
	[RelevancyPercentage] [int] NULL,
	[CompanyPercentage] [int] NULL,
	[SalesPercentage] [int] NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_Search] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Season]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Season](
	[Oid] [int] IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
	[CreatedBy] [uniqueidentifier] NULL,
	[UpdatedBy] [uniqueidentifier] NULL,
	[CreatedOn] [smalldatetime] NOT NULL,
	[UpdatedOn] [smalldatetime] NOT NULL,
	[Description] [nvarchar](100) NULL,
	[AlternativeDescription] [nvarchar](100) NULL,
	[Active] [bit] NOT NULL,
	[Ordering] [int] NOT NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_Season] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Singleton]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Singleton](
	[Oid] [int] IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
	[CreatedBy] [uniqueidentifier] NULL,
	[UpdatedBy] [uniqueidentifier] NULL,
	[CreatedOn] [smalldatetime] NOT NULL,
	[UpdatedOn] [smalldatetime] NOT NULL,
	[CompanyWebSite] [nvarchar](100) NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_Singleton] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Sleeve]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Sleeve](
	[Oid] [int] IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
	[CreatedBy] [uniqueidentifier] NULL,
	[UpdatedBy] [uniqueidentifier] NULL,
	[CreatedOn] [smalldatetime] NOT NULL,
	[UpdatedOn] [smalldatetime] NOT NULL,
	[Description] [nvarchar](100) NULL,
	[AlternativeDescription] [nvarchar](100) NULL,
	[Active] [bit] NOT NULL,
	[Ordering] [int] NOT NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_Sleeve] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[TrendTheme]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[TrendTheme](
	[Oid] [int] IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
	[CreatedBy] [uniqueidentifier] NULL,
	[UpdatedBy] [uniqueidentifier] NULL,
	[CreatedOn] [smalldatetime] NOT NULL,
	[UpdatedOn] [smalldatetime] NOT NULL,
	[Description] [nvarchar](100) NULL,
	[AlternativeDescription] [nvarchar](100) NULL,
	[Active] [bit] NOT NULL,
	[Ordering] [int] NOT NULL,
	[OptimisticLockField] [int] NULL,
 CONSTRAINT [PK_TrendTheme] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[XPObjectType]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[XPObjectType](
	[OID] [int] IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
	[TypeName] [nvarchar](254) NULL,
	[AssemblyName] [nvarchar](254) NULL,
 CONSTRAINT [PK_XPObjectType] PRIMARY KEY CLUSTERED 
(
	[OID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[XpoState]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[XpoState](
	[Oid] [uniqueidentifier] ROWGUIDCOL  NOT NULL,
	[Caption] [nvarchar](100) NULL,
	[StateMachine] [uniqueidentifier] NULL,
	[MarkerValue] [nvarchar](max) NULL,
	[TargetObjectCriteria] [nvarchar](max) NULL,
	[OptimisticLockField] [int] NULL,
	[GCRecord] [int] NULL,
 CONSTRAINT [PK_XpoState] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[XpoStateAppearance]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[XpoStateAppearance](
	[Oid] [uniqueidentifier] ROWGUIDCOL  NOT NULL,
	[State] [uniqueidentifier] NULL,
	[AppearanceItemType] [nvarchar](100) NULL,
	[Context] [nvarchar](100) NULL,
	[Criteria] [nvarchar](max) NULL,
	[Method] [nvarchar](100) NULL,
	[TargetItems] [nvarchar](100) NULL,
	[Priority] [int] NULL,
	[FontColor] [int] NULL,
	[BackColor] [int] NULL,
	[FontStyle] [int] NULL,
	[Enabled] [bit] NULL,
	[Visibility] [int] NULL,
	[OptimisticLockField] [int] NULL,
	[GCRecord] [int] NULL,
 CONSTRAINT [PK_XpoStateAppearance] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[XpoStateMachine]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[XpoStateMachine](
	[Oid] [uniqueidentifier] ROWGUIDCOL  NOT NULL,
	[Name] [nvarchar](100) NULL,
	[Active] [bit] NULL,
	[TargetObjectType] [nvarchar](max) NULL,
	[StatePropertyName] [nvarchar](100) NULL,
	[StartState] [uniqueidentifier] NULL,
	[ExpandActionsInDetailView] [bit] NULL,
	[OptimisticLockField] [int] NULL,
	[GCRecord] [int] NULL,
 CONSTRAINT [PK_XpoStateMachine] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[XpoTransition]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[XpoTransition](
	[Oid] [uniqueidentifier] ROWGUIDCOL  NOT NULL,
	[Caption] [nvarchar](100) NULL,
	[SourceState] [uniqueidentifier] NULL,
	[TargetState] [uniqueidentifier] NULL,
	[Index] [int] NULL,
	[SaveAndCloseView] [bit] NULL,
	[OptimisticLockField] [int] NULL,
	[GCRecord] [int] NULL,
 CONSTRAINT [PK_XpoTransition] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[XPWeakReference]    Script Date: 22/12/2020 3:59:26 μμ ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[XPWeakReference](
	[Oid] [uniqueidentifier] ROWGUIDCOL  NOT NULL,
	[TargetType] [int] NULL,
	[TargetKey] [nvarchar](100) NULL,
	[OptimisticLockField] [int] NULL,
	[GCRecord] [int] NULL,
	[ObjectType] [int] NULL,
 CONSTRAINT [PK_XPWeakReference] PRIMARY KEY CLUSTERED 
(
	[Oid] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Index [iCreatedBy_Adapter]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iCreatedBy_Adapter] ON [dbo].[Adapter]
(
	[CreatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Active]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Active] ON [dbo].[Adapter]
(
	[Active] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iUpdatedBy_Adapter]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iUpdatedBy_Adapter] ON [dbo].[Adapter]
(
	[UpdatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iGCRecord_Analysis]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iGCRecord_Analysis] ON [dbo].[Analysis]
(
	[GCRecord] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iAuditedObject_AuditDataItemPersistent]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iAuditedObject_AuditDataItemPersistent] ON [dbo].[AuditDataItemPersistent]
(
	[AuditedObject] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iGCRecord_AuditDataItemPersistent]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iGCRecord_AuditDataItemPersistent] ON [dbo].[AuditDataItemPersistent]
(
	[GCRecord] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iModifiedOn_AuditDataItemPersistent]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iModifiedOn_AuditDataItemPersistent] ON [dbo].[AuditDataItemPersistent]
(
	[ModifiedOn] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iNewObject_AuditDataItemPersistent]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iNewObject_AuditDataItemPersistent] ON [dbo].[AuditDataItemPersistent]
(
	[NewObject] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iOldObject_AuditDataItemPersistent]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iOldObject_AuditDataItemPersistent] ON [dbo].[AuditDataItemPersistent]
(
	[OldObject] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
SET ANSI_PADDING ON
GO
/****** Object:  Index [iOperationType_AuditDataItemPersistent]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iOperationType_AuditDataItemPersistent] ON [dbo].[AuditDataItemPersistent]
(
	[OperationType] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
SET ANSI_PADDING ON
GO
/****** Object:  Index [iUserName_AuditDataItemPersistent]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iUserName_AuditDataItemPersistent] ON [dbo].[AuditDataItemPersistent]
(
	[UserName] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iCreatedBy_Brand]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iCreatedBy_Brand] ON [dbo].[Brand]
(
	[CreatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Active]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Active] ON [dbo].[Brand]
(
	[Active] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iUpdatedBy_Brand]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iUpdatedBy_Brand] ON [dbo].[Brand]
(
	[UpdatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iCreatedBy_BusinessUnit]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iCreatedBy_BusinessUnit] ON [dbo].[BusinessUnit]
(
	[CreatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Active]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Active] ON [dbo].[BusinessUnit]
(
	[Active] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iUpdatedBy_BusinessUnit]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iUpdatedBy_BusinessUnit] ON [dbo].[BusinessUnit]
(
	[UpdatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iCreatedBy_Cluster]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iCreatedBy_Cluster] ON [dbo].[Cluster]
(
	[CreatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Active]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Active] ON [dbo].[Cluster]
(
	[Active] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Cluster_CollarDesign]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Cluster_CollarDesign] ON [dbo].[Cluster]
(
	[CollarDesign] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Cluster_Fit]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Cluster_Fit] ON [dbo].[Cluster]
(
	[Fit] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Cluster_Gender]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Cluster_Gender] ON [dbo].[Cluster]
(
	[Gender] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Cluster_Length]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Cluster_Length] ON [dbo].[Cluster]
(
	[Length] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Cluster_Lifestage]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Cluster_Lifestage] ON [dbo].[Cluster]
(
	[Lifestage] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Cluster_NeckDesign]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Cluster_NeckDesign] ON [dbo].[Cluster]
(
	[NeckDesign] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Cluster_ProductCategory]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Cluster_ProductCategory] ON [dbo].[Cluster]
(
	[ProductCategory] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Cluster_ProductSubcategory]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Cluster_ProductSubcategory] ON [dbo].[Cluster]
(
	[ProductSubcategory] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Cluster_Sleeve]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Cluster_Sleeve] ON [dbo].[Cluster]
(
	[Sleeve] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iUpdatedBy_Cluster]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iUpdatedBy_Cluster] ON [dbo].[Cluster]
(
	[UpdatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iCreatedBy_CollarDesign]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iCreatedBy_CollarDesign] ON [dbo].[CollarDesign]
(
	[CreatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Active]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Active] ON [dbo].[CollarDesign]
(
	[Active] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iUpdatedBy_CollarDesign]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iUpdatedBy_CollarDesign] ON [dbo].[CollarDesign]
(
	[UpdatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iCreatedBy_ColorRGB]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iCreatedBy_ColorRGB] ON [dbo].[ColorRGB]
(
	[CreatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Active]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Active] ON [dbo].[ColorRGB]
(
	[Active] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iUpdatedBy_ColorRGB]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iUpdatedBy_ColorRGB] ON [dbo].[ColorRGB]
(
	[UpdatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iCreatedBy_Combination]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iCreatedBy_Combination] ON [dbo].[Combination]
(
	[CreatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iUpdatedBy_Combination]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iUpdatedBy_Combination] ON [dbo].[Combination]
(
	[UpdatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iCreatedBy_CrawlSearch]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iCreatedBy_CrawlSearch] ON [dbo].[CrawlSearch]
(
	[CreatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_CrawlSearch_Adapter]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_CrawlSearch_Adapter] ON [dbo].[CrawlSearch]
(
	[Adapter] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iUpdatedBy_CrawlSearch]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iUpdatedBy_CrawlSearch] ON [dbo].[CrawlSearch]
(
	[UpdatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iCreatedBy_Dashboard]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iCreatedBy_Dashboard] ON [dbo].[Dashboard]
(
	[CreatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Active]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Active] ON [dbo].[Dashboard]
(
	[Active] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iUpdatedBy_Dashboard]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iUpdatedBy_Dashboard] ON [dbo].[Dashboard]
(
	[UpdatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iGCRecord_DashboardData]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iGCRecord_DashboardData] ON [dbo].[DashboardData]
(
	[GCRecord] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iCreatedBy_DashboardProduct]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iCreatedBy_DashboardProduct] ON [dbo].[DashboardProduct]
(
	[CreatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_DashboardProduct_Dashboard]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_DashboardProduct_Dashboard] ON [dbo].[DashboardProduct]
(
	[Dashboard] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_DashboardProduct_Product]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_DashboardProduct_Product] ON [dbo].[DashboardProduct]
(
	[Product] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iUpdatedBy_DashboardProduct]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iUpdatedBy_DashboardProduct] ON [dbo].[DashboardProduct]
(
	[UpdatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iEndOn_Event]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iEndOn_Event] ON [dbo].[Event]
(
	[EndOn] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iGCRecord_Event]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iGCRecord_Event] ON [dbo].[Event]
(
	[GCRecord] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iRecurrencePattern_Event]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iRecurrencePattern_Event] ON [dbo].[Event]
(
	[RecurrencePattern] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iStartOn_Event]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iStartOn_Event] ON [dbo].[Event]
(
	[StartOn] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iGCRecord_FileData]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iGCRecord_FileData] ON [dbo].[FileData]
(
	[GCRecord] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iGCRecord_FilteringCriterion]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iGCRecord_FilteringCriterion] ON [dbo].[FilteringCriterion]
(
	[GCRecord] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iCreatedBy_Fit]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iCreatedBy_Fit] ON [dbo].[Fit]
(
	[CreatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Active]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Active] ON [dbo].[Fit]
(
	[Active] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iUpdatedBy_Fit]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iUpdatedBy_Fit] ON [dbo].[Fit]
(
	[UpdatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iCreatedBy_Gender]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iCreatedBy_Gender] ON [dbo].[Gender]
(
	[CreatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Active]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Active] ON [dbo].[Gender]
(
	[Active] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iUpdatedBy_Gender]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iUpdatedBy_Gender] ON [dbo].[Gender]
(
	[UpdatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iGCRecord_HCategory]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iGCRecord_HCategory] ON [dbo].[HCategory]
(
	[GCRecord] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iParent_HCategory]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iParent_HCategory] ON [dbo].[HCategory]
(
	[Parent] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iCreatedBy_InspirationBackground]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iCreatedBy_InspirationBackground] ON [dbo].[InspirationBackground]
(
	[CreatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Active]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Active] ON [dbo].[InspirationBackground]
(
	[Active] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iUpdatedBy_InspirationBackground]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iUpdatedBy_InspirationBackground] ON [dbo].[InspirationBackground]
(
	[UpdatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iGCRecord_KpiDefinition]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iGCRecord_KpiDefinition] ON [dbo].[KpiDefinition]
(
	[GCRecord] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iKpiInstance_KpiDefinition]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iKpiInstance_KpiDefinition] ON [dbo].[KpiDefinition]
(
	[KpiInstance] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iKpiInstance_KpiHistoryItem]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iKpiInstance_KpiHistoryItem] ON [dbo].[KpiHistoryItem]
(
	[KpiInstance] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iGCRecord_KpiInstance]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iGCRecord_KpiInstance] ON [dbo].[KpiInstance]
(
	[GCRecord] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iKpiDefinition_KpiInstance]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iKpiDefinition_KpiInstance] ON [dbo].[KpiInstance]
(
	[KpiDefinition] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iGCRecord_KpiScorecard]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iGCRecord_KpiScorecard] ON [dbo].[KpiScorecard]
(
	[GCRecord] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iIndicators_KpiScorecardScorecards_KpiInstanceIndicators]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iIndicators_KpiScorecardScorecards_KpiInstanceIndicators] ON [dbo].[KpiScorecardScorecards_KpiInstanceIndicators]
(
	[Indicators] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iIndicatorsScorecards_KpiScorecardScorecards_KpiInstanceIndicators]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE UNIQUE NONCLUSTERED INDEX [iIndicatorsScorecards_KpiScorecardScorecards_KpiInstanceIndicators] ON [dbo].[KpiScorecardScorecards_KpiInstanceIndicators]
(
	[Indicators] ASC,
	[Scorecards] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, IGNORE_DUP_KEY = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iScorecards_KpiScorecardScorecards_KpiInstanceIndicators]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iScorecards_KpiScorecardScorecards_KpiInstanceIndicators] ON [dbo].[KpiScorecardScorecards_KpiInstanceIndicators]
(
	[Scorecards] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iCreatedBy_Length]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iCreatedBy_Length] ON [dbo].[Length]
(
	[CreatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Active]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Active] ON [dbo].[Length]
(
	[Active] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iUpdatedBy_Length]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iUpdatedBy_Length] ON [dbo].[Length]
(
	[UpdatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iCreatedBy_LifeStage]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iCreatedBy_LifeStage] ON [dbo].[LifeStage]
(
	[CreatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Active]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Active] ON [dbo].[LifeStage]
(
	[Active] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iUpdatedBy_LifeStage]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iUpdatedBy_LifeStage] ON [dbo].[LifeStage]
(
	[UpdatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iCreatedBy_Manufacturer]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iCreatedBy_Manufacturer] ON [dbo].[Manufacturer]
(
	[CreatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Active]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Active] ON [dbo].[Manufacturer]
(
	[Active] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iUpdatedBy_Manufacturer]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iUpdatedBy_Manufacturer] ON [dbo].[Manufacturer]
(
	[UpdatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iGCRecord_ModelDifference]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iGCRecord_ModelDifference] ON [dbo].[ModelDifference]
(
	[GCRecord] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iGCRecord_ModelDifferenceAspect]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iGCRecord_ModelDifferenceAspect] ON [dbo].[ModelDifferenceAspect]
(
	[GCRecord] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iOwner_ModelDifferenceAspect]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iOwner_ModelDifferenceAspect] ON [dbo].[ModelDifferenceAspect]
(
	[Owner] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iCreatedBy_NeckDesign]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iCreatedBy_NeckDesign] ON [dbo].[NeckDesign]
(
	[CreatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Active]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Active] ON [dbo].[NeckDesign]
(
	[Active] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iUpdatedBy_NeckDesign]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iUpdatedBy_NeckDesign] ON [dbo].[NeckDesign]
(
	[UpdatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iGCRecord_PermissionPolicyActionPermissionObject]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iGCRecord_PermissionPolicyActionPermissionObject] ON [dbo].[PermissionPolicyActionPermissionObject]
(
	[GCRecord] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iRole_PermissionPolicyActionPermissionObject]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iRole_PermissionPolicyActionPermissionObject] ON [dbo].[PermissionPolicyActionPermissionObject]
(
	[Role] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iGCRecord_PermissionPolicyMemberPermissionsObject]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iGCRecord_PermissionPolicyMemberPermissionsObject] ON [dbo].[PermissionPolicyMemberPermissionsObject]
(
	[GCRecord] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iTypePermissionObject_PermissionPolicyMemberPermissionsObject]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iTypePermissionObject_PermissionPolicyMemberPermissionsObject] ON [dbo].[PermissionPolicyMemberPermissionsObject]
(
	[TypePermissionObject] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iGCRecord_PermissionPolicyNavigationPermissionsObject]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iGCRecord_PermissionPolicyNavigationPermissionsObject] ON [dbo].[PermissionPolicyNavigationPermissionsObject]
(
	[GCRecord] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iRole_PermissionPolicyNavigationPermissionsObject]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iRole_PermissionPolicyNavigationPermissionsObject] ON [dbo].[PermissionPolicyNavigationPermissionsObject]
(
	[Role] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iGCRecord_PermissionPolicyObjectPermissionsObject]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iGCRecord_PermissionPolicyObjectPermissionsObject] ON [dbo].[PermissionPolicyObjectPermissionsObject]
(
	[GCRecord] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iTypePermissionObject_PermissionPolicyObjectPermissionsObject]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iTypePermissionObject_PermissionPolicyObjectPermissionsObject] ON [dbo].[PermissionPolicyObjectPermissionsObject]
(
	[TypePermissionObject] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iGCRecord_PermissionPolicyRole]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iGCRecord_PermissionPolicyRole] ON [dbo].[PermissionPolicyRole]
(
	[GCRecord] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iObjectType_PermissionPolicyRole]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iObjectType_PermissionPolicyRole] ON [dbo].[PermissionPolicyRole]
(
	[ObjectType] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iGCRecord_PermissionPolicyTypePermissionsObject]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iGCRecord_PermissionPolicyTypePermissionsObject] ON [dbo].[PermissionPolicyTypePermissionsObject]
(
	[GCRecord] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iRole_PermissionPolicyTypePermissionsObject]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iRole_PermissionPolicyTypePermissionsObject] ON [dbo].[PermissionPolicyTypePermissionsObject]
(
	[Role] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iGCRecord_PermissionPolicyUser]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iGCRecord_PermissionPolicyUser] ON [dbo].[PermissionPolicyUser]
(
	[GCRecord] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iRoles_PermissionPolicyUserUsers_PermissionPolicyRoleRoles]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iRoles_PermissionPolicyUserUsers_PermissionPolicyRoleRoles] ON [dbo].[PermissionPolicyUserUsers_PermissionPolicyRoleRoles]
(
	[Roles] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iRolesUsers_PermissionPolicyUserUsers_PermissionPolicyRoleRoles]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE UNIQUE NONCLUSTERED INDEX [iRolesUsers_PermissionPolicyUserUsers_PermissionPolicyRoleRoles] ON [dbo].[PermissionPolicyUserUsers_PermissionPolicyRoleRoles]
(
	[Roles] ASC,
	[Users] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, IGNORE_DUP_KEY = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iUsers_PermissionPolicyUserUsers_PermissionPolicyRoleRoles]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iUsers_PermissionPolicyUserUsers_PermissionPolicyRoleRoles] ON [dbo].[PermissionPolicyUserUsers_PermissionPolicyRoleRoles]
(
	[Users] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iCreatedBy_Product]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iCreatedBy_Product] ON [dbo].[Product]
(
	[CreatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Active]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Active] ON [dbo].[Product]
(
	[Active] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Product_Adapter]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Product_Adapter] ON [dbo].[Product]
(
	[Adapter] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Product_Brand]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Product_Brand] ON [dbo].[Product]
(
	[Brand] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Product_BusinessUnit]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Product_BusinessUnit] ON [dbo].[Product]
(
	[BusinessUnit] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Product_Cluster]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Product_Cluster] ON [dbo].[Product]
(
	[Cluster] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Product_CollarDesign]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Product_CollarDesign] ON [dbo].[Product]
(
	[CollarDesign] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Product_FinancialCluster]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Product_FinancialCluster] ON [dbo].[Product]
(
	[FinancialCluster] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Product_Fit]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Product_Fit] ON [dbo].[Product]
(
	[Fit] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Product_Gender]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Product_Gender] ON [dbo].[Product]
(
	[Gender] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Product_InspirationBackground]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Product_InspirationBackground] ON [dbo].[Product]
(
	[InspirationBackground] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Product_Length]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Product_Length] ON [dbo].[Product]
(
	[Length] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Product_LifeStage]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Product_LifeStage] ON [dbo].[Product]
(
	[LifeStage] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Product_NeckDesign]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Product_NeckDesign] ON [dbo].[Product]
(
	[NeckDesign] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Product_ProductCategory]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Product_ProductCategory] ON [dbo].[Product]
(
	[ProductCategory] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Product_ProductionManufacturer]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Product_ProductionManufacturer] ON [dbo].[Product]
(
	[ProductionManufacturer] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Product_ProductSubcategory]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Product_ProductSubcategory] ON [dbo].[Product]
(
	[ProductSubcategory] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Product_SampleManufacturer]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Product_SampleManufacturer] ON [dbo].[Product]
(
	[SampleManufacturer] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Product_Season]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Product_Season] ON [dbo].[Product]
(
	[Season] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Product_Sleeve]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Product_Sleeve] ON [dbo].[Product]
(
	[Sleeve] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Product_TrendTheme]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Product_TrendTheme] ON [dbo].[Product]
(
	[TrendTheme] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iUpdatedBy_Product]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iUpdatedBy_Product] ON [dbo].[Product]
(
	[UpdatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iCreatedBy_ProductCategory]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iCreatedBy_ProductCategory] ON [dbo].[ProductCategory]
(
	[CreatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Active]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Active] ON [dbo].[ProductCategory]
(
	[Active] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iUpdatedBy_ProductCategory]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iUpdatedBy_ProductCategory] ON [dbo].[ProductCategory]
(
	[UpdatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iCreatedBy_ProductColor]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iCreatedBy_ProductColor] ON [dbo].[ProductColor]
(
	[CreatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_ProductColor_ColorRGB]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_ProductColor_ColorRGB] ON [dbo].[ProductColor]
(
	[ColorRGB] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_ProductColor_Product]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_ProductColor_Product] ON [dbo].[ProductColor]
(
	[Product] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iUpdatedBy_ProductColor]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iUpdatedBy_ProductColor] ON [dbo].[ProductColor]
(
	[UpdatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iCreatedBy_ProductHistory]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iCreatedBy_ProductHistory] ON [dbo].[ProductHistory]
(
	[CreatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_ProductHistory_Product]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_ProductHistory_Product] ON [dbo].[ProductHistory]
(
	[Product] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iUpdatedBy_ProductHistory]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iUpdatedBy_ProductHistory] ON [dbo].[ProductHistory]
(
	[UpdatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iCreatedBy_ProductSubcategory]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iCreatedBy_ProductSubcategory] ON [dbo].[ProductSubcategory]
(
	[CreatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Active]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Active] ON [dbo].[ProductSubcategory]
(
	[Active] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_ProductSubcategory_ProductCategory]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_ProductSubcategory_ProductCategory] ON [dbo].[ProductSubcategory]
(
	[ProductCategory] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iUpdatedBy_ProductSubcategory]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iUpdatedBy_ProductSubcategory] ON [dbo].[ProductSubcategory]
(
	[UpdatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iGCRecord_ReportDataV2]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iGCRecord_ReportDataV2] ON [dbo].[ReportDataV2]
(
	[GCRecord] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iGCRecord_Resource]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iGCRecord_Resource] ON [dbo].[Resource]
(
	[GCRecord] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iEvents_ResourceResources_EventEvents]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iEvents_ResourceResources_EventEvents] ON [dbo].[ResourceResources_EventEvents]
(
	[Events] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iEventsResources_ResourceResources_EventEvents]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE UNIQUE NONCLUSTERED INDEX [iEventsResources_ResourceResources_EventEvents] ON [dbo].[ResourceResources_EventEvents]
(
	[Events] ASC,
	[Resources] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, IGNORE_DUP_KEY = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iResources_ResourceResources_EventEvents]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iResources_ResourceResources_EventEvents] ON [dbo].[ResourceResources_EventEvents]
(
	[Resources] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iCreatedBy_Restriction]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iCreatedBy_Restriction] ON [dbo].[Restriction]
(
	[CreatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iUpdatedBy_Restriction]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iUpdatedBy_Restriction] ON [dbo].[Restriction]
(
	[UpdatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iCreatedBy_Result]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iCreatedBy_Result] ON [dbo].[Result]
(
	[CreatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Result_Product]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Result_Product] ON [dbo].[Result]
(
	[Product] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Result_Search]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Result_Search] ON [dbo].[Result]
(
	[Search] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iUpdatedBy_Result]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iUpdatedBy_Result] ON [dbo].[Result]
(
	[UpdatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iCreatedBy_Search]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iCreatedBy_Search] ON [dbo].[Search]
(
	[CreatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Search_Season]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Search_Season] ON [dbo].[Search]
(
	[Season] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iUpdatedBy_Search]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iUpdatedBy_Search] ON [dbo].[Search]
(
	[UpdatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iCreatedBy_Season]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iCreatedBy_Season] ON [dbo].[Season]
(
	[CreatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Active]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Active] ON [dbo].[Season]
(
	[Active] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iUpdatedBy_Season]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iUpdatedBy_Season] ON [dbo].[Season]
(
	[UpdatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iCreatedBy_Singleton]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iCreatedBy_Singleton] ON [dbo].[Singleton]
(
	[CreatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iUpdatedBy_Singleton]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iUpdatedBy_Singleton] ON [dbo].[Singleton]
(
	[UpdatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iCreatedBy_Sleeve]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iCreatedBy_Sleeve] ON [dbo].[Sleeve]
(
	[CreatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Active]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Active] ON [dbo].[Sleeve]
(
	[Active] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iUpdatedBy_Sleeve]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iUpdatedBy_Sleeve] ON [dbo].[Sleeve]
(
	[UpdatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iCreatedBy_TrendTheme]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iCreatedBy_TrendTheme] ON [dbo].[TrendTheme]
(
	[CreatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [IDX_Active]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [IDX_Active] ON [dbo].[TrendTheme]
(
	[Active] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iUpdatedBy_TrendTheme]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iUpdatedBy_TrendTheme] ON [dbo].[TrendTheme]
(
	[UpdatedBy] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
SET ANSI_PADDING ON
GO
/****** Object:  Index [iTypeName_XPObjectType]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE UNIQUE NONCLUSTERED INDEX [iTypeName_XPObjectType] ON [dbo].[XPObjectType]
(
	[TypeName] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, IGNORE_DUP_KEY = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iGCRecord_XpoState]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iGCRecord_XpoState] ON [dbo].[XpoState]
(
	[GCRecord] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iStateMachine_XpoState]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iStateMachine_XpoState] ON [dbo].[XpoState]
(
	[StateMachine] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iGCRecord_XpoStateAppearance]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iGCRecord_XpoStateAppearance] ON [dbo].[XpoStateAppearance]
(
	[GCRecord] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iState_XpoStateAppearance]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iState_XpoStateAppearance] ON [dbo].[XpoStateAppearance]
(
	[State] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iGCRecord_XpoStateMachine]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iGCRecord_XpoStateMachine] ON [dbo].[XpoStateMachine]
(
	[GCRecord] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iStartState_XpoStateMachine]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iStartState_XpoStateMachine] ON [dbo].[XpoStateMachine]
(
	[StartState] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iGCRecord_XpoTransition]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iGCRecord_XpoTransition] ON [dbo].[XpoTransition]
(
	[GCRecord] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iSourceState_XpoTransition]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iSourceState_XpoTransition] ON [dbo].[XpoTransition]
(
	[SourceState] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iTargetState_XpoTransition]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iTargetState_XpoTransition] ON [dbo].[XpoTransition]
(
	[TargetState] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iGCRecord_XPWeakReference]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iGCRecord_XPWeakReference] ON [dbo].[XPWeakReference]
(
	[GCRecord] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iObjectType_XPWeakReference]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iObjectType_XPWeakReference] ON [dbo].[XPWeakReference]
(
	[ObjectType] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
/****** Object:  Index [iTargetType_XPWeakReference]    Script Date: 22/12/2020 3:59:26 μμ ******/
CREATE NONCLUSTERED INDEX [iTargetType_XPWeakReference] ON [dbo].[XPWeakReference]
(
	[TargetType] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
ALTER TABLE [dbo].[Adapter] ADD  DEFAULT (getutcdate()) FOR [CreatedOn]
GO
ALTER TABLE [dbo].[Adapter] ADD  DEFAULT (getutcdate()) FOR [UpdatedOn]
GO
ALTER TABLE [dbo].[Adapter] ADD  DEFAULT ((1)) FOR [Active]
GO
ALTER TABLE [dbo].[Adapter] ADD  DEFAULT ((0)) FOR [Ordering]
GO
ALTER TABLE [dbo].[Brand] ADD  DEFAULT (getutcdate()) FOR [CreatedOn]
GO
ALTER TABLE [dbo].[Brand] ADD  DEFAULT (getutcdate()) FOR [UpdatedOn]
GO
ALTER TABLE [dbo].[Brand] ADD  DEFAULT ((1)) FOR [Active]
GO
ALTER TABLE [dbo].[Brand] ADD  DEFAULT ((0)) FOR [Ordering]
GO
ALTER TABLE [dbo].[BusinessUnit] ADD  DEFAULT (getutcdate()) FOR [CreatedOn]
GO
ALTER TABLE [dbo].[BusinessUnit] ADD  DEFAULT (getutcdate()) FOR [UpdatedOn]
GO
ALTER TABLE [dbo].[BusinessUnit] ADD  DEFAULT ((1)) FOR [Active]
GO
ALTER TABLE [dbo].[BusinessUnit] ADD  DEFAULT ((0)) FOR [Ordering]
GO
ALTER TABLE [dbo].[Cluster] ADD  DEFAULT (getutcdate()) FOR [CreatedOn]
GO
ALTER TABLE [dbo].[Cluster] ADD  DEFAULT (getutcdate()) FOR [UpdatedOn]
GO
ALTER TABLE [dbo].[Cluster] ADD  DEFAULT ((1)) FOR [Active]
GO
ALTER TABLE [dbo].[Cluster] ADD  DEFAULT ((0)) FOR [Ordering]
GO
ALTER TABLE [dbo].[CollarDesign] ADD  DEFAULT (getutcdate()) FOR [CreatedOn]
GO
ALTER TABLE [dbo].[CollarDesign] ADD  DEFAULT (getutcdate()) FOR [UpdatedOn]
GO
ALTER TABLE [dbo].[CollarDesign] ADD  DEFAULT ((1)) FOR [Active]
GO
ALTER TABLE [dbo].[CollarDesign] ADD  DEFAULT ((0)) FOR [Ordering]
GO
ALTER TABLE [dbo].[ColorRGB] ADD  DEFAULT (getutcdate()) FOR [CreatedOn]
GO
ALTER TABLE [dbo].[ColorRGB] ADD  DEFAULT (getutcdate()) FOR [UpdatedOn]
GO
ALTER TABLE [dbo].[ColorRGB] ADD  DEFAULT ((1)) FOR [Active]
GO
ALTER TABLE [dbo].[ColorRGB] ADD  DEFAULT ((0)) FOR [Ordering]
GO
ALTER TABLE [dbo].[Combination] ADD  DEFAULT (getutcdate()) FOR [CreatedOn]
GO
ALTER TABLE [dbo].[Combination] ADD  DEFAULT (getutcdate()) FOR [UpdatedOn]
GO
ALTER TABLE [dbo].[CrawlSearch] ADD  DEFAULT (getutcdate()) FOR [CreatedOn]
GO
ALTER TABLE [dbo].[CrawlSearch] ADD  DEFAULT (getutcdate()) FOR [UpdatedOn]
GO
ALTER TABLE [dbo].[Dashboard] ADD  DEFAULT (getutcdate()) FOR [CreatedOn]
GO
ALTER TABLE [dbo].[Dashboard] ADD  DEFAULT (getutcdate()) FOR [UpdatedOn]
GO
ALTER TABLE [dbo].[Dashboard] ADD  DEFAULT ((1)) FOR [Active]
GO
ALTER TABLE [dbo].[Dashboard] ADD  DEFAULT ((0)) FOR [Ordering]
GO
ALTER TABLE [dbo].[DashboardProduct] ADD  DEFAULT (getutcdate()) FOR [CreatedOn]
GO
ALTER TABLE [dbo].[DashboardProduct] ADD  DEFAULT (getutcdate()) FOR [UpdatedOn]
GO
ALTER TABLE [dbo].[Fit] ADD  DEFAULT (getutcdate()) FOR [CreatedOn]
GO
ALTER TABLE [dbo].[Fit] ADD  DEFAULT (getutcdate()) FOR [UpdatedOn]
GO
ALTER TABLE [dbo].[Fit] ADD  DEFAULT ((1)) FOR [Active]
GO
ALTER TABLE [dbo].[Fit] ADD  DEFAULT ((0)) FOR [Ordering]
GO
ALTER TABLE [dbo].[Gender] ADD  DEFAULT (getutcdate()) FOR [CreatedOn]
GO
ALTER TABLE [dbo].[Gender] ADD  DEFAULT (getutcdate()) FOR [UpdatedOn]
GO
ALTER TABLE [dbo].[Gender] ADD  DEFAULT ((1)) FOR [Active]
GO
ALTER TABLE [dbo].[Gender] ADD  DEFAULT ((0)) FOR [Ordering]
GO
ALTER TABLE [dbo].[InspirationBackground] ADD  DEFAULT (getutcdate()) FOR [CreatedOn]
GO
ALTER TABLE [dbo].[InspirationBackground] ADD  DEFAULT (getutcdate()) FOR [UpdatedOn]
GO
ALTER TABLE [dbo].[InspirationBackground] ADD  DEFAULT ((1)) FOR [Active]
GO
ALTER TABLE [dbo].[InspirationBackground] ADD  DEFAULT ((0)) FOR [Ordering]
GO
ALTER TABLE [dbo].[Length] ADD  DEFAULT (getutcdate()) FOR [CreatedOn]
GO
ALTER TABLE [dbo].[Length] ADD  DEFAULT (getutcdate()) FOR [UpdatedOn]
GO
ALTER TABLE [dbo].[Length] ADD  DEFAULT ((1)) FOR [Active]
GO
ALTER TABLE [dbo].[Length] ADD  DEFAULT ((0)) FOR [Ordering]
GO
ALTER TABLE [dbo].[LifeStage] ADD  DEFAULT (getutcdate()) FOR [CreatedOn]
GO
ALTER TABLE [dbo].[LifeStage] ADD  DEFAULT (getutcdate()) FOR [UpdatedOn]
GO
ALTER TABLE [dbo].[LifeStage] ADD  DEFAULT ((1)) FOR [Active]
GO
ALTER TABLE [dbo].[LifeStage] ADD  DEFAULT ((0)) FOR [Ordering]
GO
ALTER TABLE [dbo].[Manufacturer] ADD  DEFAULT (getutcdate()) FOR [CreatedOn]
GO
ALTER TABLE [dbo].[Manufacturer] ADD  DEFAULT (getutcdate()) FOR [UpdatedOn]
GO
ALTER TABLE [dbo].[Manufacturer] ADD  DEFAULT ((1)) FOR [Active]
GO
ALTER TABLE [dbo].[Manufacturer] ADD  DEFAULT ((0)) FOR [Ordering]
GO
ALTER TABLE [dbo].[NeckDesign] ADD  DEFAULT (getutcdate()) FOR [CreatedOn]
GO
ALTER TABLE [dbo].[NeckDesign] ADD  DEFAULT (getutcdate()) FOR [UpdatedOn]
GO
ALTER TABLE [dbo].[NeckDesign] ADD  DEFAULT ((1)) FOR [Active]
GO
ALTER TABLE [dbo].[NeckDesign] ADD  DEFAULT ((0)) FOR [Ordering]
GO
ALTER TABLE [dbo].[Product] ADD  DEFAULT (getutcdate()) FOR [CreatedOn]
GO
ALTER TABLE [dbo].[Product] ADD  DEFAULT (getutcdate()) FOR [UpdatedOn]
GO
ALTER TABLE [dbo].[Product] ADD  DEFAULT ((1)) FOR [Active]
GO
ALTER TABLE [dbo].[Product] ADD  DEFAULT ((0)) FOR [Ordering]
GO
ALTER TABLE [dbo].[ProductCategory] ADD  DEFAULT (getutcdate()) FOR [CreatedOn]
GO
ALTER TABLE [dbo].[ProductCategory] ADD  DEFAULT (getutcdate()) FOR [UpdatedOn]
GO
ALTER TABLE [dbo].[ProductCategory] ADD  DEFAULT ((1)) FOR [Active]
GO
ALTER TABLE [dbo].[ProductCategory] ADD  DEFAULT ((0)) FOR [Ordering]
GO
ALTER TABLE [dbo].[ProductColor] ADD  DEFAULT (getutcdate()) FOR [CreatedOn]
GO
ALTER TABLE [dbo].[ProductColor] ADD  DEFAULT (getutcdate()) FOR [UpdatedOn]
GO
ALTER TABLE [dbo].[ProductHistory] ADD  DEFAULT (getutcdate()) FOR [CreatedOn]
GO
ALTER TABLE [dbo].[ProductHistory] ADD  DEFAULT (getutcdate()) FOR [UpdatedOn]
GO
ALTER TABLE [dbo].[ProductSubcategory] ADD  DEFAULT (getutcdate()) FOR [CreatedOn]
GO
ALTER TABLE [dbo].[ProductSubcategory] ADD  DEFAULT (getutcdate()) FOR [UpdatedOn]
GO
ALTER TABLE [dbo].[ProductSubcategory] ADD  DEFAULT ((1)) FOR [Active]
GO
ALTER TABLE [dbo].[ProductSubcategory] ADD  DEFAULT ((0)) FOR [Ordering]
GO
ALTER TABLE [dbo].[Restriction] ADD  DEFAULT (getutcdate()) FOR [CreatedOn]
GO
ALTER TABLE [dbo].[Restriction] ADD  DEFAULT (getutcdate()) FOR [UpdatedOn]
GO
ALTER TABLE [dbo].[Result] ADD  DEFAULT (getutcdate()) FOR [CreatedOn]
GO
ALTER TABLE [dbo].[Result] ADD  DEFAULT (getutcdate()) FOR [UpdatedOn]
GO
ALTER TABLE [dbo].[Result] ADD  DEFAULT ((0)) FOR [Clicked]
GO
ALTER TABLE [dbo].[Result] ADD  DEFAULT ((0)) FOR [IsFavorite]
GO
ALTER TABLE [dbo].[Result] ADD  DEFAULT ((0)) FOR [IsIrrelevant]
GO
ALTER TABLE [dbo].[Search] ADD  DEFAULT (getutcdate()) FOR [CreatedOn]
GO
ALTER TABLE [dbo].[Search] ADD  DEFAULT (getutcdate()) FOR [UpdatedOn]
GO
ALTER TABLE [dbo].[Season] ADD  DEFAULT (getutcdate()) FOR [CreatedOn]
GO
ALTER TABLE [dbo].[Season] ADD  DEFAULT (getutcdate()) FOR [UpdatedOn]
GO
ALTER TABLE [dbo].[Season] ADD  DEFAULT ((1)) FOR [Active]
GO
ALTER TABLE [dbo].[Season] ADD  DEFAULT ((0)) FOR [Ordering]
GO
ALTER TABLE [dbo].[Singleton] ADD  DEFAULT (getutcdate()) FOR [CreatedOn]
GO
ALTER TABLE [dbo].[Singleton] ADD  DEFAULT (getutcdate()) FOR [UpdatedOn]
GO
ALTER TABLE [dbo].[Sleeve] ADD  DEFAULT (getutcdate()) FOR [CreatedOn]
GO
ALTER TABLE [dbo].[Sleeve] ADD  DEFAULT (getutcdate()) FOR [UpdatedOn]
GO
ALTER TABLE [dbo].[Sleeve] ADD  DEFAULT ((1)) FOR [Active]
GO
ALTER TABLE [dbo].[Sleeve] ADD  DEFAULT ((0)) FOR [Ordering]
GO
ALTER TABLE [dbo].[TrendTheme] ADD  DEFAULT (getutcdate()) FOR [CreatedOn]
GO
ALTER TABLE [dbo].[TrendTheme] ADD  DEFAULT (getutcdate()) FOR [UpdatedOn]
GO
ALTER TABLE [dbo].[TrendTheme] ADD  DEFAULT ((1)) FOR [Active]
GO
ALTER TABLE [dbo].[TrendTheme] ADD  DEFAULT ((0)) FOR [Ordering]
GO
ALTER TABLE [dbo].[Adapter]  WITH NOCHECK ADD  CONSTRAINT [FK_Adapter_CreatedBy] FOREIGN KEY([CreatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Adapter] CHECK CONSTRAINT [FK_Adapter_CreatedBy]
GO
ALTER TABLE [dbo].[Adapter]  WITH NOCHECK ADD  CONSTRAINT [FK_Adapter_UpdatedBy] FOREIGN KEY([UpdatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Adapter] CHECK CONSTRAINT [FK_Adapter_UpdatedBy]
GO
ALTER TABLE [dbo].[AuditDataItemPersistent]  WITH NOCHECK ADD  CONSTRAINT [FK_AuditDataItemPersistent_AuditedObject] FOREIGN KEY([AuditedObject])
REFERENCES [dbo].[AuditedObjectWeakReference] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[AuditDataItemPersistent] CHECK CONSTRAINT [FK_AuditDataItemPersistent_AuditedObject]
GO
ALTER TABLE [dbo].[AuditDataItemPersistent]  WITH NOCHECK ADD  CONSTRAINT [FK_AuditDataItemPersistent_NewObject] FOREIGN KEY([NewObject])
REFERENCES [dbo].[XPWeakReference] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[AuditDataItemPersistent] CHECK CONSTRAINT [FK_AuditDataItemPersistent_NewObject]
GO
ALTER TABLE [dbo].[AuditDataItemPersistent]  WITH NOCHECK ADD  CONSTRAINT [FK_AuditDataItemPersistent_OldObject] FOREIGN KEY([OldObject])
REFERENCES [dbo].[XPWeakReference] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[AuditDataItemPersistent] CHECK CONSTRAINT [FK_AuditDataItemPersistent_OldObject]
GO
ALTER TABLE [dbo].[AuditedObjectWeakReference]  WITH NOCHECK ADD  CONSTRAINT [FK_AuditedObjectWeakReference_Oid] FOREIGN KEY([Oid])
REFERENCES [dbo].[XPWeakReference] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[AuditedObjectWeakReference] CHECK CONSTRAINT [FK_AuditedObjectWeakReference_Oid]
GO
ALTER TABLE [dbo].[Brand]  WITH NOCHECK ADD  CONSTRAINT [FK_Brand_CreatedBy] FOREIGN KEY([CreatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Brand] CHECK CONSTRAINT [FK_Brand_CreatedBy]
GO
ALTER TABLE [dbo].[Brand]  WITH NOCHECK ADD  CONSTRAINT [FK_Brand_UpdatedBy] FOREIGN KEY([UpdatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Brand] CHECK CONSTRAINT [FK_Brand_UpdatedBy]
GO
ALTER TABLE [dbo].[BusinessUnit]  WITH NOCHECK ADD  CONSTRAINT [FK_BusinessUnit_CreatedBy] FOREIGN KEY([CreatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[BusinessUnit] CHECK CONSTRAINT [FK_BusinessUnit_CreatedBy]
GO
ALTER TABLE [dbo].[BusinessUnit]  WITH NOCHECK ADD  CONSTRAINT [FK_BusinessUnit_UpdatedBy] FOREIGN KEY([UpdatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[BusinessUnit] CHECK CONSTRAINT [FK_BusinessUnit_UpdatedBy]
GO
ALTER TABLE [dbo].[Cluster]  WITH NOCHECK ADD  CONSTRAINT [FK_Cluster_CollarDesign] FOREIGN KEY([CollarDesign])
REFERENCES [dbo].[CollarDesign] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Cluster] CHECK CONSTRAINT [FK_Cluster_CollarDesign]
GO
ALTER TABLE [dbo].[Cluster]  WITH NOCHECK ADD  CONSTRAINT [FK_Cluster_CreatedBy] FOREIGN KEY([CreatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Cluster] CHECK CONSTRAINT [FK_Cluster_CreatedBy]
GO
ALTER TABLE [dbo].[Cluster]  WITH NOCHECK ADD  CONSTRAINT [FK_Cluster_Fit] FOREIGN KEY([Fit])
REFERENCES [dbo].[Fit] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Cluster] CHECK CONSTRAINT [FK_Cluster_Fit]
GO
ALTER TABLE [dbo].[Cluster]  WITH NOCHECK ADD  CONSTRAINT [FK_Cluster_Gender] FOREIGN KEY([Gender])
REFERENCES [dbo].[Gender] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Cluster] CHECK CONSTRAINT [FK_Cluster_Gender]
GO
ALTER TABLE [dbo].[Cluster]  WITH NOCHECK ADD  CONSTRAINT [FK_Cluster_Length] FOREIGN KEY([Length])
REFERENCES [dbo].[Length] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Cluster] CHECK CONSTRAINT [FK_Cluster_Length]
GO
ALTER TABLE [dbo].[Cluster]  WITH NOCHECK ADD  CONSTRAINT [FK_Cluster_Lifestage] FOREIGN KEY([Lifestage])
REFERENCES [dbo].[LifeStage] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Cluster] CHECK CONSTRAINT [FK_Cluster_Lifestage]
GO
ALTER TABLE [dbo].[Cluster]  WITH NOCHECK ADD  CONSTRAINT [FK_Cluster_NeckDesign] FOREIGN KEY([NeckDesign])
REFERENCES [dbo].[NeckDesign] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Cluster] CHECK CONSTRAINT [FK_Cluster_NeckDesign]
GO
ALTER TABLE [dbo].[Cluster]  WITH NOCHECK ADD  CONSTRAINT [FK_Cluster_ProductCategory] FOREIGN KEY([ProductCategory])
REFERENCES [dbo].[ProductCategory] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Cluster] CHECK CONSTRAINT [FK_Cluster_ProductCategory]
GO
ALTER TABLE [dbo].[Cluster]  WITH NOCHECK ADD  CONSTRAINT [FK_Cluster_ProductSubcategory] FOREIGN KEY([ProductSubcategory])
REFERENCES [dbo].[ProductSubcategory] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Cluster] CHECK CONSTRAINT [FK_Cluster_ProductSubcategory]
GO
ALTER TABLE [dbo].[Cluster]  WITH NOCHECK ADD  CONSTRAINT [FK_Cluster_Sleeve] FOREIGN KEY([Sleeve])
REFERENCES [dbo].[Sleeve] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Cluster] CHECK CONSTRAINT [FK_Cluster_Sleeve]
GO
ALTER TABLE [dbo].[Cluster]  WITH NOCHECK ADD  CONSTRAINT [FK_Cluster_UpdatedBy] FOREIGN KEY([UpdatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Cluster] CHECK CONSTRAINT [FK_Cluster_UpdatedBy]
GO
ALTER TABLE [dbo].[CollarDesign]  WITH NOCHECK ADD  CONSTRAINT [FK_CollarDesign_CreatedBy] FOREIGN KEY([CreatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[CollarDesign] CHECK CONSTRAINT [FK_CollarDesign_CreatedBy]
GO
ALTER TABLE [dbo].[CollarDesign]  WITH NOCHECK ADD  CONSTRAINT [FK_CollarDesign_UpdatedBy] FOREIGN KEY([UpdatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[CollarDesign] CHECK CONSTRAINT [FK_CollarDesign_UpdatedBy]
GO
ALTER TABLE [dbo].[ColorRGB]  WITH NOCHECK ADD  CONSTRAINT [FK_ColorRGB_CreatedBy] FOREIGN KEY([CreatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[ColorRGB] CHECK CONSTRAINT [FK_ColorRGB_CreatedBy]
GO
ALTER TABLE [dbo].[ColorRGB]  WITH NOCHECK ADD  CONSTRAINT [FK_ColorRGB_UpdatedBy] FOREIGN KEY([UpdatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[ColorRGB] CHECK CONSTRAINT [FK_ColorRGB_UpdatedBy]
GO
ALTER TABLE [dbo].[Combination]  WITH NOCHECK ADD  CONSTRAINT [FK_Combination_CreatedBy] FOREIGN KEY([CreatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Combination] CHECK CONSTRAINT [FK_Combination_CreatedBy]
GO
ALTER TABLE [dbo].[Combination]  WITH NOCHECK ADD  CONSTRAINT [FK_Combination_UpdatedBy] FOREIGN KEY([UpdatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Combination] CHECK CONSTRAINT [FK_Combination_UpdatedBy]
GO
ALTER TABLE [dbo].[CrawlSearch]  WITH NOCHECK ADD  CONSTRAINT [FK_CrawlSearch_Adapter] FOREIGN KEY([Adapter])
REFERENCES [dbo].[Adapter] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[CrawlSearch] CHECK CONSTRAINT [FK_CrawlSearch_Adapter]
GO
ALTER TABLE [dbo].[CrawlSearch]  WITH NOCHECK ADD  CONSTRAINT [FK_CrawlSearch_CreatedBy] FOREIGN KEY([CreatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[CrawlSearch] CHECK CONSTRAINT [FK_CrawlSearch_CreatedBy]
GO
ALTER TABLE [dbo].[CrawlSearch]  WITH NOCHECK ADD  CONSTRAINT [FK_CrawlSearch_UpdatedBy] FOREIGN KEY([UpdatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[CrawlSearch] CHECK CONSTRAINT [FK_CrawlSearch_UpdatedBy]
GO
ALTER TABLE [dbo].[Dashboard]  WITH NOCHECK ADD  CONSTRAINT [FK_Dashboard_CreatedBy] FOREIGN KEY([CreatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Dashboard] CHECK CONSTRAINT [FK_Dashboard_CreatedBy]
GO
ALTER TABLE [dbo].[Dashboard]  WITH NOCHECK ADD  CONSTRAINT [FK_Dashboard_UpdatedBy] FOREIGN KEY([UpdatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Dashboard] CHECK CONSTRAINT [FK_Dashboard_UpdatedBy]
GO
ALTER TABLE [dbo].[DashboardProduct]  WITH NOCHECK ADD  CONSTRAINT [FK_DashboardProduct_CreatedBy] FOREIGN KEY([CreatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[DashboardProduct] CHECK CONSTRAINT [FK_DashboardProduct_CreatedBy]
GO
ALTER TABLE [dbo].[DashboardProduct]  WITH NOCHECK ADD  CONSTRAINT [FK_DashboardProduct_Dashboard] FOREIGN KEY([Dashboard])
REFERENCES [dbo].[Dashboard] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[DashboardProduct] CHECK CONSTRAINT [FK_DashboardProduct_Dashboard]
GO
ALTER TABLE [dbo].[DashboardProduct]  WITH NOCHECK ADD  CONSTRAINT [FK_DashboardProduct_Product] FOREIGN KEY([Product])
REFERENCES [dbo].[Product] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[DashboardProduct] CHECK CONSTRAINT [FK_DashboardProduct_Product]
GO
ALTER TABLE [dbo].[DashboardProduct]  WITH NOCHECK ADD  CONSTRAINT [FK_DashboardProduct_UpdatedBy] FOREIGN KEY([UpdatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[DashboardProduct] CHECK CONSTRAINT [FK_DashboardProduct_UpdatedBy]
GO
ALTER TABLE [dbo].[Event]  WITH NOCHECK ADD  CONSTRAINT [FK_Event_RecurrencePattern] FOREIGN KEY([RecurrencePattern])
REFERENCES [dbo].[Event] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Event] CHECK CONSTRAINT [FK_Event_RecurrencePattern]
GO
ALTER TABLE [dbo].[Fit]  WITH NOCHECK ADD  CONSTRAINT [FK_Fit_CreatedBy] FOREIGN KEY([CreatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Fit] CHECK CONSTRAINT [FK_Fit_CreatedBy]
GO
ALTER TABLE [dbo].[Fit]  WITH NOCHECK ADD  CONSTRAINT [FK_Fit_UpdatedBy] FOREIGN KEY([UpdatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Fit] CHECK CONSTRAINT [FK_Fit_UpdatedBy]
GO
ALTER TABLE [dbo].[Gender]  WITH NOCHECK ADD  CONSTRAINT [FK_Gender_CreatedBy] FOREIGN KEY([CreatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Gender] CHECK CONSTRAINT [FK_Gender_CreatedBy]
GO
ALTER TABLE [dbo].[Gender]  WITH NOCHECK ADD  CONSTRAINT [FK_Gender_UpdatedBy] FOREIGN KEY([UpdatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Gender] CHECK CONSTRAINT [FK_Gender_UpdatedBy]
GO
ALTER TABLE [dbo].[HCategory]  WITH NOCHECK ADD  CONSTRAINT [FK_HCategory_Parent] FOREIGN KEY([Parent])
REFERENCES [dbo].[HCategory] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[HCategory] CHECK CONSTRAINT [FK_HCategory_Parent]
GO
ALTER TABLE [dbo].[InspirationBackground]  WITH NOCHECK ADD  CONSTRAINT [FK_InspirationBackground_CreatedBy] FOREIGN KEY([CreatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[InspirationBackground] CHECK CONSTRAINT [FK_InspirationBackground_CreatedBy]
GO
ALTER TABLE [dbo].[InspirationBackground]  WITH NOCHECK ADD  CONSTRAINT [FK_InspirationBackground_UpdatedBy] FOREIGN KEY([UpdatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[InspirationBackground] CHECK CONSTRAINT [FK_InspirationBackground_UpdatedBy]
GO
ALTER TABLE [dbo].[KpiDefinition]  WITH NOCHECK ADD  CONSTRAINT [FK_KpiDefinition_KpiInstance] FOREIGN KEY([KpiInstance])
REFERENCES [dbo].[KpiInstance] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[KpiDefinition] CHECK CONSTRAINT [FK_KpiDefinition_KpiInstance]
GO
ALTER TABLE [dbo].[KpiHistoryItem]  WITH NOCHECK ADD  CONSTRAINT [FK_KpiHistoryItem_KpiInstance] FOREIGN KEY([KpiInstance])
REFERENCES [dbo].[KpiInstance] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[KpiHistoryItem] CHECK CONSTRAINT [FK_KpiHistoryItem_KpiInstance]
GO
ALTER TABLE [dbo].[KpiInstance]  WITH NOCHECK ADD  CONSTRAINT [FK_KpiInstance_KpiDefinition] FOREIGN KEY([KpiDefinition])
REFERENCES [dbo].[KpiDefinition] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[KpiInstance] CHECK CONSTRAINT [FK_KpiInstance_KpiDefinition]
GO
ALTER TABLE [dbo].[KpiScorecardScorecards_KpiInstanceIndicators]  WITH NOCHECK ADD  CONSTRAINT [FK_KpiScorecardScorecards_KpiInstanceIndicators_Indicators] FOREIGN KEY([Indicators])
REFERENCES [dbo].[KpiInstance] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[KpiScorecardScorecards_KpiInstanceIndicators] CHECK CONSTRAINT [FK_KpiScorecardScorecards_KpiInstanceIndicators_Indicators]
GO
ALTER TABLE [dbo].[KpiScorecardScorecards_KpiInstanceIndicators]  WITH NOCHECK ADD  CONSTRAINT [FK_KpiScorecardScorecards_KpiInstanceIndicators_Scorecards] FOREIGN KEY([Scorecards])
REFERENCES [dbo].[KpiScorecard] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[KpiScorecardScorecards_KpiInstanceIndicators] CHECK CONSTRAINT [FK_KpiScorecardScorecards_KpiInstanceIndicators_Scorecards]
GO
ALTER TABLE [dbo].[Length]  WITH NOCHECK ADD  CONSTRAINT [FK_Length_CreatedBy] FOREIGN KEY([CreatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Length] CHECK CONSTRAINT [FK_Length_CreatedBy]
GO
ALTER TABLE [dbo].[Length]  WITH NOCHECK ADD  CONSTRAINT [FK_Length_UpdatedBy] FOREIGN KEY([UpdatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Length] CHECK CONSTRAINT [FK_Length_UpdatedBy]
GO
ALTER TABLE [dbo].[LifeStage]  WITH NOCHECK ADD  CONSTRAINT [FK_LifeStage_CreatedBy] FOREIGN KEY([CreatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[LifeStage] CHECK CONSTRAINT [FK_LifeStage_CreatedBy]
GO
ALTER TABLE [dbo].[LifeStage]  WITH NOCHECK ADD  CONSTRAINT [FK_LifeStage_UpdatedBy] FOREIGN KEY([UpdatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[LifeStage] CHECK CONSTRAINT [FK_LifeStage_UpdatedBy]
GO
ALTER TABLE [dbo].[Manufacturer]  WITH NOCHECK ADD  CONSTRAINT [FK_Manufacturer_CreatedBy] FOREIGN KEY([CreatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Manufacturer] CHECK CONSTRAINT [FK_Manufacturer_CreatedBy]
GO
ALTER TABLE [dbo].[Manufacturer]  WITH NOCHECK ADD  CONSTRAINT [FK_Manufacturer_UpdatedBy] FOREIGN KEY([UpdatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Manufacturer] CHECK CONSTRAINT [FK_Manufacturer_UpdatedBy]
GO
ALTER TABLE [dbo].[ModelDifferenceAspect]  WITH NOCHECK ADD  CONSTRAINT [FK_ModelDifferenceAspect_Owner] FOREIGN KEY([Owner])
REFERENCES [dbo].[ModelDifference] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[ModelDifferenceAspect] CHECK CONSTRAINT [FK_ModelDifferenceAspect_Owner]
GO
ALTER TABLE [dbo].[NeckDesign]  WITH NOCHECK ADD  CONSTRAINT [FK_NeckDesign_CreatedBy] FOREIGN KEY([CreatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[NeckDesign] CHECK CONSTRAINT [FK_NeckDesign_CreatedBy]
GO
ALTER TABLE [dbo].[NeckDesign]  WITH NOCHECK ADD  CONSTRAINT [FK_NeckDesign_UpdatedBy] FOREIGN KEY([UpdatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[NeckDesign] CHECK CONSTRAINT [FK_NeckDesign_UpdatedBy]
GO
ALTER TABLE [dbo].[PermissionPolicyActionPermissionObject]  WITH NOCHECK ADD  CONSTRAINT [FK_PermissionPolicyActionPermissionObject_Role] FOREIGN KEY([Role])
REFERENCES [dbo].[PermissionPolicyRole] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[PermissionPolicyActionPermissionObject] CHECK CONSTRAINT [FK_PermissionPolicyActionPermissionObject_Role]
GO
ALTER TABLE [dbo].[PermissionPolicyMemberPermissionsObject]  WITH NOCHECK ADD  CONSTRAINT [FK_PermissionPolicyMemberPermissionsObject_TypePermissionObject] FOREIGN KEY([TypePermissionObject])
REFERENCES [dbo].[PermissionPolicyTypePermissionsObject] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[PermissionPolicyMemberPermissionsObject] CHECK CONSTRAINT [FK_PermissionPolicyMemberPermissionsObject_TypePermissionObject]
GO
ALTER TABLE [dbo].[PermissionPolicyNavigationPermissionsObject]  WITH NOCHECK ADD  CONSTRAINT [FK_PermissionPolicyNavigationPermissionsObject_Role] FOREIGN KEY([Role])
REFERENCES [dbo].[PermissionPolicyRole] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[PermissionPolicyNavigationPermissionsObject] CHECK CONSTRAINT [FK_PermissionPolicyNavigationPermissionsObject_Role]
GO
ALTER TABLE [dbo].[PermissionPolicyObjectPermissionsObject]  WITH NOCHECK ADD  CONSTRAINT [FK_PermissionPolicyObjectPermissionsObject_TypePermissionObject] FOREIGN KEY([TypePermissionObject])
REFERENCES [dbo].[PermissionPolicyTypePermissionsObject] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[PermissionPolicyObjectPermissionsObject] CHECK CONSTRAINT [FK_PermissionPolicyObjectPermissionsObject_TypePermissionObject]
GO
ALTER TABLE [dbo].[PermissionPolicyRole]  WITH NOCHECK ADD  CONSTRAINT [FK_PermissionPolicyRole_ObjectType] FOREIGN KEY([ObjectType])
REFERENCES [dbo].[XPObjectType] ([OID])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[PermissionPolicyRole] CHECK CONSTRAINT [FK_PermissionPolicyRole_ObjectType]
GO
ALTER TABLE [dbo].[PermissionPolicyTypePermissionsObject]  WITH NOCHECK ADD  CONSTRAINT [FK_PermissionPolicyTypePermissionsObject_Role] FOREIGN KEY([Role])
REFERENCES [dbo].[PermissionPolicyRole] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[PermissionPolicyTypePermissionsObject] CHECK CONSTRAINT [FK_PermissionPolicyTypePermissionsObject_Role]
GO
ALTER TABLE [dbo].[PermissionPolicyUserUsers_PermissionPolicyRoleRoles]  WITH NOCHECK ADD  CONSTRAINT [FK_PermissionPolicyUserUsers_PermissionPolicyRoleRoles_Roles] FOREIGN KEY([Roles])
REFERENCES [dbo].[PermissionPolicyRole] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[PermissionPolicyUserUsers_PermissionPolicyRoleRoles] CHECK CONSTRAINT [FK_PermissionPolicyUserUsers_PermissionPolicyRoleRoles_Roles]
GO
ALTER TABLE [dbo].[PermissionPolicyUserUsers_PermissionPolicyRoleRoles]  WITH NOCHECK ADD  CONSTRAINT [FK_PermissionPolicyUserUsers_PermissionPolicyRoleRoles_Users] FOREIGN KEY([Users])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[PermissionPolicyUserUsers_PermissionPolicyRoleRoles] CHECK CONSTRAINT [FK_PermissionPolicyUserUsers_PermissionPolicyRoleRoles_Users]
GO
ALTER TABLE [dbo].[Product]  WITH NOCHECK ADD  CONSTRAINT [FK_Product_Adapter] FOREIGN KEY([Adapter])
REFERENCES [dbo].[Adapter] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Product] CHECK CONSTRAINT [FK_Product_Adapter]
GO
ALTER TABLE [dbo].[Product]  WITH NOCHECK ADD  CONSTRAINT [FK_Product_Brand] FOREIGN KEY([Brand])
REFERENCES [dbo].[Brand] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Product] CHECK CONSTRAINT [FK_Product_Brand]
GO
ALTER TABLE [dbo].[Product]  WITH NOCHECK ADD  CONSTRAINT [FK_Product_BusinessUnit] FOREIGN KEY([BusinessUnit])
REFERENCES [dbo].[BusinessUnit] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Product] CHECK CONSTRAINT [FK_Product_BusinessUnit]
GO
ALTER TABLE [dbo].[Product]  WITH NOCHECK ADD  CONSTRAINT [FK_Product_Cluster] FOREIGN KEY([Cluster])
REFERENCES [dbo].[Cluster] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Product] CHECK CONSTRAINT [FK_Product_Cluster]
GO
ALTER TABLE [dbo].[Product]  WITH NOCHECK ADD  CONSTRAINT [FK_Product_CollarDesign] FOREIGN KEY([CollarDesign])
REFERENCES [dbo].[CollarDesign] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Product] CHECK CONSTRAINT [FK_Product_CollarDesign]
GO
ALTER TABLE [dbo].[Product]  WITH NOCHECK ADD  CONSTRAINT [FK_Product_CreatedBy] FOREIGN KEY([CreatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Product] CHECK CONSTRAINT [FK_Product_CreatedBy]
GO
ALTER TABLE [dbo].[Product]  WITH NOCHECK ADD  CONSTRAINT [FK_Product_FinancialCluster] FOREIGN KEY([FinancialCluster])
REFERENCES [dbo].[Cluster] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Product] CHECK CONSTRAINT [FK_Product_FinancialCluster]
GO
ALTER TABLE [dbo].[Product]  WITH NOCHECK ADD  CONSTRAINT [FK_Product_Fit] FOREIGN KEY([Fit])
REFERENCES [dbo].[Fit] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Product] CHECK CONSTRAINT [FK_Product_Fit]
GO
ALTER TABLE [dbo].[Product]  WITH NOCHECK ADD  CONSTRAINT [FK_Product_Gender] FOREIGN KEY([Gender])
REFERENCES [dbo].[Gender] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Product] CHECK CONSTRAINT [FK_Product_Gender]
GO
ALTER TABLE [dbo].[Product]  WITH NOCHECK ADD  CONSTRAINT [FK_Product_InspirationBackground] FOREIGN KEY([InspirationBackground])
REFERENCES [dbo].[InspirationBackground] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Product] CHECK CONSTRAINT [FK_Product_InspirationBackground]
GO
ALTER TABLE [dbo].[Product]  WITH NOCHECK ADD  CONSTRAINT [FK_Product_Length] FOREIGN KEY([Length])
REFERENCES [dbo].[Length] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Product] CHECK CONSTRAINT [FK_Product_Length]
GO
ALTER TABLE [dbo].[Product]  WITH NOCHECK ADD  CONSTRAINT [FK_Product_LifeStage] FOREIGN KEY([LifeStage])
REFERENCES [dbo].[LifeStage] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Product] CHECK CONSTRAINT [FK_Product_LifeStage]
GO
ALTER TABLE [dbo].[Product]  WITH NOCHECK ADD  CONSTRAINT [FK_Product_NeckDesign] FOREIGN KEY([NeckDesign])
REFERENCES [dbo].[NeckDesign] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Product] CHECK CONSTRAINT [FK_Product_NeckDesign]
GO
ALTER TABLE [dbo].[Product]  WITH NOCHECK ADD  CONSTRAINT [FK_Product_ProductCategory] FOREIGN KEY([ProductCategory])
REFERENCES [dbo].[ProductCategory] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Product] CHECK CONSTRAINT [FK_Product_ProductCategory]
GO
ALTER TABLE [dbo].[Product]  WITH NOCHECK ADD  CONSTRAINT [FK_Product_ProductionManufacturer] FOREIGN KEY([ProductionManufacturer])
REFERENCES [dbo].[Manufacturer] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Product] CHECK CONSTRAINT [FK_Product_ProductionManufacturer]
GO
ALTER TABLE [dbo].[Product]  WITH NOCHECK ADD  CONSTRAINT [FK_Product_ProductSubcategory] FOREIGN KEY([ProductSubcategory])
REFERENCES [dbo].[ProductSubcategory] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Product] CHECK CONSTRAINT [FK_Product_ProductSubcategory]
GO
ALTER TABLE [dbo].[Product]  WITH NOCHECK ADD  CONSTRAINT [FK_Product_SampleManufacturer] FOREIGN KEY([SampleManufacturer])
REFERENCES [dbo].[Manufacturer] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Product] CHECK CONSTRAINT [FK_Product_SampleManufacturer]
GO
ALTER TABLE [dbo].[Product]  WITH NOCHECK ADD  CONSTRAINT [FK_Product_Season] FOREIGN KEY([Season])
REFERENCES [dbo].[Season] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Product] CHECK CONSTRAINT [FK_Product_Season]
GO
ALTER TABLE [dbo].[Product]  WITH NOCHECK ADD  CONSTRAINT [FK_Product_Sleeve] FOREIGN KEY([Sleeve])
REFERENCES [dbo].[Sleeve] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Product] CHECK CONSTRAINT [FK_Product_Sleeve]
GO
ALTER TABLE [dbo].[Product]  WITH NOCHECK ADD  CONSTRAINT [FK_Product_TrendTheme] FOREIGN KEY([TrendTheme])
REFERENCES [dbo].[TrendTheme] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Product] CHECK CONSTRAINT [FK_Product_TrendTheme]
GO
ALTER TABLE [dbo].[Product]  WITH NOCHECK ADD  CONSTRAINT [FK_Product_UpdatedBy] FOREIGN KEY([UpdatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Product] CHECK CONSTRAINT [FK_Product_UpdatedBy]
GO
ALTER TABLE [dbo].[ProductCategory]  WITH NOCHECK ADD  CONSTRAINT [FK_ProductCategory_CreatedBy] FOREIGN KEY([CreatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[ProductCategory] CHECK CONSTRAINT [FK_ProductCategory_CreatedBy]
GO
ALTER TABLE [dbo].[ProductCategory]  WITH NOCHECK ADD  CONSTRAINT [FK_ProductCategory_UpdatedBy] FOREIGN KEY([UpdatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[ProductCategory] CHECK CONSTRAINT [FK_ProductCategory_UpdatedBy]
GO
ALTER TABLE [dbo].[ProductColor]  WITH NOCHECK ADD  CONSTRAINT [FK_ProductColor_ColorRGB] FOREIGN KEY([ColorRGB])
REFERENCES [dbo].[ColorRGB] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[ProductColor] CHECK CONSTRAINT [FK_ProductColor_ColorRGB]
GO
ALTER TABLE [dbo].[ProductColor]  WITH NOCHECK ADD  CONSTRAINT [FK_ProductColor_CreatedBy] FOREIGN KEY([CreatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[ProductColor] CHECK CONSTRAINT [FK_ProductColor_CreatedBy]
GO
ALTER TABLE [dbo].[ProductColor]  WITH NOCHECK ADD  CONSTRAINT [FK_ProductColor_Product] FOREIGN KEY([Product])
REFERENCES [dbo].[Product] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[ProductColor] CHECK CONSTRAINT [FK_ProductColor_Product]
GO
ALTER TABLE [dbo].[ProductColor]  WITH NOCHECK ADD  CONSTRAINT [FK_ProductColor_UpdatedBy] FOREIGN KEY([UpdatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[ProductColor] CHECK CONSTRAINT [FK_ProductColor_UpdatedBy]
GO
ALTER TABLE [dbo].[ProductHistory]  WITH NOCHECK ADD  CONSTRAINT [FK_ProductHistory_CreatedBy] FOREIGN KEY([CreatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[ProductHistory] CHECK CONSTRAINT [FK_ProductHistory_CreatedBy]
GO
ALTER TABLE [dbo].[ProductHistory]  WITH NOCHECK ADD  CONSTRAINT [FK_ProductHistory_Product] FOREIGN KEY([Product])
REFERENCES [dbo].[Product] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[ProductHistory] CHECK CONSTRAINT [FK_ProductHistory_Product]
GO
ALTER TABLE [dbo].[ProductHistory]  WITH NOCHECK ADD  CONSTRAINT [FK_ProductHistory_UpdatedBy] FOREIGN KEY([UpdatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[ProductHistory] CHECK CONSTRAINT [FK_ProductHistory_UpdatedBy]
GO
ALTER TABLE [dbo].[ProductSubcategory]  WITH NOCHECK ADD  CONSTRAINT [FK_ProductSubcategory_CreatedBy] FOREIGN KEY([CreatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[ProductSubcategory] CHECK CONSTRAINT [FK_ProductSubcategory_CreatedBy]
GO
ALTER TABLE [dbo].[ProductSubcategory]  WITH NOCHECK ADD  CONSTRAINT [FK_ProductSubcategory_ProductCategory] FOREIGN KEY([ProductCategory])
REFERENCES [dbo].[ProductCategory] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[ProductSubcategory] CHECK CONSTRAINT [FK_ProductSubcategory_ProductCategory]
GO
ALTER TABLE [dbo].[ProductSubcategory]  WITH NOCHECK ADD  CONSTRAINT [FK_ProductSubcategory_UpdatedBy] FOREIGN KEY([UpdatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[ProductSubcategory] CHECK CONSTRAINT [FK_ProductSubcategory_UpdatedBy]
GO
ALTER TABLE [dbo].[ResourceResources_EventEvents]  WITH NOCHECK ADD  CONSTRAINT [FK_ResourceResources_EventEvents_Events] FOREIGN KEY([Events])
REFERENCES [dbo].[Event] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[ResourceResources_EventEvents] CHECK CONSTRAINT [FK_ResourceResources_EventEvents_Events]
GO
ALTER TABLE [dbo].[ResourceResources_EventEvents]  WITH NOCHECK ADD  CONSTRAINT [FK_ResourceResources_EventEvents_Resources] FOREIGN KEY([Resources])
REFERENCES [dbo].[Resource] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[ResourceResources_EventEvents] CHECK CONSTRAINT [FK_ResourceResources_EventEvents_Resources]
GO
ALTER TABLE [dbo].[Restriction]  WITH NOCHECK ADD  CONSTRAINT [FK_Restriction_CreatedBy] FOREIGN KEY([CreatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Restriction] CHECK CONSTRAINT [FK_Restriction_CreatedBy]
GO
ALTER TABLE [dbo].[Restriction]  WITH NOCHECK ADD  CONSTRAINT [FK_Restriction_UpdatedBy] FOREIGN KEY([UpdatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Restriction] CHECK CONSTRAINT [FK_Restriction_UpdatedBy]
GO
ALTER TABLE [dbo].[Result]  WITH NOCHECK ADD  CONSTRAINT [FK_Result_CreatedBy] FOREIGN KEY([CreatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Result] CHECK CONSTRAINT [FK_Result_CreatedBy]
GO
ALTER TABLE [dbo].[Result]  WITH NOCHECK ADD  CONSTRAINT [FK_Result_Product] FOREIGN KEY([Product])
REFERENCES [dbo].[Product] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Result] CHECK CONSTRAINT [FK_Result_Product]
GO
ALTER TABLE [dbo].[Result]  WITH NOCHECK ADD  CONSTRAINT [FK_Result_Search] FOREIGN KEY([Search])
REFERENCES [dbo].[Search] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Result] CHECK CONSTRAINT [FK_Result_Search]
GO
ALTER TABLE [dbo].[Result]  WITH NOCHECK ADD  CONSTRAINT [FK_Result_UpdatedBy] FOREIGN KEY([UpdatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Result] CHECK CONSTRAINT [FK_Result_UpdatedBy]
GO
ALTER TABLE [dbo].[Search]  WITH NOCHECK ADD  CONSTRAINT [FK_Search_CreatedBy] FOREIGN KEY([CreatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Search] CHECK CONSTRAINT [FK_Search_CreatedBy]
GO
ALTER TABLE [dbo].[Search]  WITH NOCHECK ADD  CONSTRAINT [FK_Search_Season] FOREIGN KEY([Season])
REFERENCES [dbo].[Season] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Search] CHECK CONSTRAINT [FK_Search_Season]
GO
ALTER TABLE [dbo].[Search]  WITH NOCHECK ADD  CONSTRAINT [FK_Search_UpdatedBy] FOREIGN KEY([UpdatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Search] CHECK CONSTRAINT [FK_Search_UpdatedBy]
GO
ALTER TABLE [dbo].[Season]  WITH NOCHECK ADD  CONSTRAINT [FK_Season_CreatedBy] FOREIGN KEY([CreatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Season] CHECK CONSTRAINT [FK_Season_CreatedBy]
GO
ALTER TABLE [dbo].[Season]  WITH NOCHECK ADD  CONSTRAINT [FK_Season_UpdatedBy] FOREIGN KEY([UpdatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Season] CHECK CONSTRAINT [FK_Season_UpdatedBy]
GO
ALTER TABLE [dbo].[Singleton]  WITH NOCHECK ADD  CONSTRAINT [FK_Singleton_CreatedBy] FOREIGN KEY([CreatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Singleton] CHECK CONSTRAINT [FK_Singleton_CreatedBy]
GO
ALTER TABLE [dbo].[Singleton]  WITH NOCHECK ADD  CONSTRAINT [FK_Singleton_UpdatedBy] FOREIGN KEY([UpdatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Singleton] CHECK CONSTRAINT [FK_Singleton_UpdatedBy]
GO
ALTER TABLE [dbo].[Sleeve]  WITH NOCHECK ADD  CONSTRAINT [FK_Sleeve_CreatedBy] FOREIGN KEY([CreatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Sleeve] CHECK CONSTRAINT [FK_Sleeve_CreatedBy]
GO
ALTER TABLE [dbo].[Sleeve]  WITH NOCHECK ADD  CONSTRAINT [FK_Sleeve_UpdatedBy] FOREIGN KEY([UpdatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[Sleeve] CHECK CONSTRAINT [FK_Sleeve_UpdatedBy]
GO
ALTER TABLE [dbo].[TrendTheme]  WITH NOCHECK ADD  CONSTRAINT [FK_TrendTheme_CreatedBy] FOREIGN KEY([CreatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[TrendTheme] CHECK CONSTRAINT [FK_TrendTheme_CreatedBy]
GO
ALTER TABLE [dbo].[TrendTheme]  WITH NOCHECK ADD  CONSTRAINT [FK_TrendTheme_UpdatedBy] FOREIGN KEY([UpdatedBy])
REFERENCES [dbo].[PermissionPolicyUser] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[TrendTheme] CHECK CONSTRAINT [FK_TrendTheme_UpdatedBy]
GO
ALTER TABLE [dbo].[XpoState]  WITH NOCHECK ADD  CONSTRAINT [FK_XpoState_StateMachine] FOREIGN KEY([StateMachine])
REFERENCES [dbo].[XpoStateMachine] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[XpoState] CHECK CONSTRAINT [FK_XpoState_StateMachine]
GO
ALTER TABLE [dbo].[XpoStateAppearance]  WITH NOCHECK ADD  CONSTRAINT [FK_XpoStateAppearance_State] FOREIGN KEY([State])
REFERENCES [dbo].[XpoState] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[XpoStateAppearance] CHECK CONSTRAINT [FK_XpoStateAppearance_State]
GO
ALTER TABLE [dbo].[XpoStateMachine]  WITH NOCHECK ADD  CONSTRAINT [FK_XpoStateMachine_StartState] FOREIGN KEY([StartState])
REFERENCES [dbo].[XpoState] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[XpoStateMachine] CHECK CONSTRAINT [FK_XpoStateMachine_StartState]
GO
ALTER TABLE [dbo].[XpoTransition]  WITH NOCHECK ADD  CONSTRAINT [FK_XpoTransition_SourceState] FOREIGN KEY([SourceState])
REFERENCES [dbo].[XpoState] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[XpoTransition] CHECK CONSTRAINT [FK_XpoTransition_SourceState]
GO
ALTER TABLE [dbo].[XpoTransition]  WITH NOCHECK ADD  CONSTRAINT [FK_XpoTransition_TargetState] FOREIGN KEY([TargetState])
REFERENCES [dbo].[XpoState] ([Oid])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[XpoTransition] CHECK CONSTRAINT [FK_XpoTransition_TargetState]
GO
ALTER TABLE [dbo].[XPWeakReference]  WITH NOCHECK ADD  CONSTRAINT [FK_XPWeakReference_ObjectType] FOREIGN KEY([ObjectType])
REFERENCES [dbo].[XPObjectType] ([OID])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[XPWeakReference] CHECK CONSTRAINT [FK_XPWeakReference_ObjectType]
GO
ALTER TABLE [dbo].[XPWeakReference]  WITH NOCHECK ADD  CONSTRAINT [FK_XPWeakReference_TargetType] FOREIGN KEY([TargetType])
REFERENCES [dbo].[XPObjectType] ([OID])
NOT FOR REPLICATION 
GO
ALTER TABLE [dbo].[XPWeakReference] CHECK CONSTRAINT [FK_XPWeakReference_TargetType]
GO
USE [master]
GO
ALTER DATABASE [S4F_clone] SET  READ_WRITE 
GO
